# clients/BrainTumorSegmentation3dClient/ClientImpl.py
# FULL UPDATED (A->Z) to match server strategy:
# - FEDAVG: raw weights, no encode, UsedSparsity=0, UsedClusters=0
# - SKH: sparsity + clusters
# - SH: sparsity only, UsedClusters="-"
# - KH: clusters only, UsedSparsity=0
# - Writes ClientCommReport.csv consistently

import logging
import torch
import flwr as fl
import numpy as np
import os
import csv
import pickle
from pathlib import Path
from dotenv import load_dotenv

from clients.BrainTumorSegmentation3dClient.utils import (
    train,
    test,
    post_transforms_with_invert as post_transforms,
)

from utils import encode_weights, decode_weights, estimate_comm_bits_theoretical


# ----------------- Model hash (debug) ----------------- #
def hash_model(model):
    import hashlib
    total = b"".join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    return hashlib.md5(total).hexdigest()


# ----------------- Load .env variables ----------------- #
dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

ENCODING_MODE = os.getenv("ENCODING_MODE", "FEDAVG").upper()  # FEDAVG / SKH / SH / KH
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "3"))

SPARSITY_CONV = float(os.getenv("SPARSITY_CONV", "0.60"))
SPARSITY_LINEAR = float(os.getenv("SPARSITY_LINEAR", "0.60"))
SPARSITY_BIAS = float(os.getenv("SPARSITY_BIAS", "0.0"))
SPARSITY_BN = float(os.getenv("SPARSITY_BN", "0.0"))

ENV_SPARSITY_CFG = {
    "Conv": SPARSITY_CONV,
    "Linear": SPARSITY_LINEAR,
    "Bias": SPARSITY_BIAS,
    "BatchNorm": SPARSITY_BN,
}


def active_knobs(mode: str, n_clusters: int, env_sparsity_cfg: dict):
    """Return active sparsity_cfg, used_sparsity (string), used_clusters (string) following your rules."""
    mode = (mode or "").upper()

    if mode == "FEDAVG":
        return {
            "mode": "FEDAVG",
            "sparsity_cfg": {"Conv": 0.0, "Linear": 0.0, "Bias": 0.0, "BatchNorm": 0.0},
            "used_sparsity": "0",
            "used_clusters": "0",
            "clusters_on": False,
            "sparsity_on": False,
        }

    if mode == "KH":
        return {
            "mode": "KH",
            "sparsity_cfg": {"Conv": 0.0, "Linear": 0.0, "Bias": 0.0, "BatchNorm": 0.0},
            "used_sparsity": "0",
            "used_clusters": str(int(n_clusters)),
            "clusters_on": True,
            "sparsity_on": False,
        }

    if mode == "SH":
        # clusters not used
        return {
            "mode": "SH",
            "sparsity_cfg": dict(env_sparsity_cfg),
            "used_sparsity": str(float(env_sparsity_cfg.get("Conv", 0.0))),
            "used_clusters": "-",
            "clusters_on": False,
            "sparsity_on": True,
        }

    # default SKH
    return {
        "mode": "SKH",
        "sparsity_cfg": dict(env_sparsity_cfg),
        "used_sparsity": str(float(env_sparsity_cfg.get("Conv", 0.0))),
        "used_clusters": str(int(n_clusters)),
        "clusters_on": True,
        "sparsity_on": True,
    }


class BrainTumorSegmentation3dClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        loss_function,
        optimizer,
        scaler,
        lr_scheduler,
        local_epochs,
        device,
        dice_metric,
        post_transforms,
        log_file_path=None,
        metrics_csv_path=None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.dice_metric = dice_metric
        self.post_transforms = post_transforms
        self.metrics_csv_path = metrics_csv_path

        # Comm CSV path in same folder as metrics
        self.comm_csv_path = None
        if metrics_csv_path is not None:
            base_dir = os.path.dirname(metrics_csv_path)
            os.makedirs(base_dir, exist_ok=True)
            self.comm_csv_path = os.path.join(base_dir, "ClientCommReport.csv")

        # Logger
        self.logger = logging.getLogger(f"client_logger_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        if log_file_path:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
            if not self.logger.hasHandlers():
                self.logger.addHandler(file_handler)

        knobs = active_knobs(ENCODING_MODE, N_CLUSTERS, ENV_SPARSITY_CFG)

        self.logger.info(f"Model initialized on device: {self.device}")
        self.logger.info(
            f"[CLIENT INIT] Mode={knobs['mode']} "
            f"| Sparsity={'ON' if knobs['sparsity_on'] else 'OFF'} (UsedSparsity={knobs['used_sparsity']}) "
            f"| Clusters={'ON' if knobs['clusters_on'] else 'OFF'} (UsedClusters={knobs['used_clusters']}) "
            f"| EnvSparsityCfg={ENV_SPARSITY_CFG} | EnvNClusters={N_CLUSTERS}"
        )

    # -----------------------------
    # Comm CSV writer
    # -----------------------------
    def _write_comm_csv(self, rnd: int, mode: str, used_sparsity: str, used_clusters: str,
                        baseline_bits: float, compressed_bits: float):
        if self.comm_csv_path is None:
            return

        csv_exists = os.path.exists(self.comm_csv_path)
        cr = (baseline_bits / compressed_bits) if compressed_bits > 0 else 1.0

        try:
            with open(self.comm_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    writer.writerow(
                        [
                            "Round",
                            "EncodingMode",
                            "UsedSparsity",
                            "UsedClusters",
                            "BaselineBits",
                            "CompressedBits",
                            "CompressionRatio",
                        ]
                    )
                writer.writerow(
                    [
                        int(rnd),
                        str(mode),
                        str(used_sparsity),
                        str(used_clusters),
                        int(baseline_bits),
                        int(compressed_bits),
                        float(cr),
                    ]
                )
        except Exception as e:
            self.logger.error(f"[ClientCommReport CSV ERROR] {e}")

    # -----------------------------
    # Parameters -> send to server
    # -----------------------------
    def get_parameters(self, config=None, server_round=None):
        state_dict = self.model.state_dict()
        sorted_keys = sorted(state_dict.keys())
        weights = [state_dict[k].detach().cpu().numpy() for k in sorted_keys]

        rnd = int(server_round) if server_round is not None else -1
        knobs = active_knobs(ENCODING_MODE, N_CLUSTERS, ENV_SPARSITY_CFG)

        baseline_bits = float(sum(int(w.size) * 32 for w in weights))

        # --------------------------
        # FEDAVG: raw weights
        # --------------------------
        if knobs["mode"] == "FEDAVG":
            compressed_bits = baseline_bits
            self.logger.info(
                f"[COMM][CLIENT] Round={rnd} Mode=FEDAVG UsedSparsity=0 UsedClusters=0 "
                f"BaselineBits={baseline_bits:.1f} CompressedBits={compressed_bits:.1f} CR=1.000"
            )
            self._write_comm_csv(rnd, "FEDAVG", "0", "0", baseline_bits, compressed_bits)

            # return raw float arrays
            return weights

        # --------------------------
        # SKH / SH / KH: encoded path
        # --------------------------
        # NOTE: For TRUE SH/KH semantics, your root-level utils.py must implement mode logic.
        # Here we enforce the active knobs:
        # - KH: sparsity_cfg=0
        # - SH: used_clusters="-", but we still pass n_clusters=1 to avoid accidental KMeans usage
        # - SKH: pass N_CLUSTERS

        n_clusters_to_use = N_CLUSTERS if knobs["clusters_on"] else 1

        encoded = encode_weights(
            weights,
            sparsity_cfg=knobs["sparsity_cfg"],
            n_clusters=n_clusters_to_use,
        )

        compressed_bits = float(estimate_comm_bits_theoretical(encoded, knobs["sparsity_cfg"]))
        cr = (baseline_bits / compressed_bits) if compressed_bits > 0 else 1.0

        self.logger.info(
            f"[COMM][CLIENT] Round={rnd} Mode={knobs['mode']} "
            f"UsedSparsity={knobs['used_sparsity']} UsedClusters={knobs['used_clusters']} "
            f"BaselineBits={baseline_bits:.1f} CompressedBits={compressed_bits:.1f} CR={cr:.3f}"
        )
        self._write_comm_csv(
            rnd,
            knobs["mode"],
            knobs["used_sparsity"],
            knobs["used_clusters"],
            baseline_bits,
            compressed_bits,
        )

        # optional debug
        print("[CLIENT DEBUG] Model hash:", hash_model(self.model))

        # serialize for server decode (uint8 pickles)
        def serialize_component(obj):
            return np.frombuffer(pickle.dumps(obj), dtype=np.uint8)

        return [
            serialize_component(encoded["centroids"]),
            serialize_component(encoded["encoded_labels"]),
            serialize_component(encoded["label_dicts"]),
            serialize_component(encoded.get("address_table", [])),
            serialize_component(encoded["shapes"]),
            serialize_component(encoded["mode"]),
            serialize_component(encoded["sparse_indices"]),
            serialize_component(sorted_keys),
        ]

    # -----------------------------
    # Receive from server -> set model
    # -----------------------------
    def set_parameters(self, parameters):
        def deserialize(arr):
            return pickle.loads(arr.tobytes())

        # Uncompressed path (FEDAVG or server round 0)
        if isinstance(parameters, list) and all(p.dtype != np.uint8 for p in parameters):
            state_dict = self.model.state_dict()
            new_state_dict = {
                k: torch.tensor(v, dtype=torch.float32).reshape(state_dict[k].shape)
                for k, v in zip(sorted(state_dict.keys()), parameters)
            }
            self.model.load_state_dict(new_state_dict, strict=True)
            self.logger.info("[DEBUG] Loaded uncompressed weights.")
            return

        # Compressed path
        encoded_dict = {
            "centroids": deserialize(parameters[0]),
            "encoded_labels": deserialize(parameters[1]),
            "label_dicts": deserialize(parameters[2]),
            "address_table": deserialize(parameters[3]),
            "shapes": deserialize(parameters[4]),
            "mode": deserialize(parameters[5]),
            "sparse_indices": deserialize(parameters[6]),
        }
        layer_names = deserialize(parameters[7])

        decoded_weights = decode_weights(encoded_dict)

        state_dict = self.model.state_dict()
        new_state_dict = {
            k: torch.tensor(v, dtype=torch.float32).reshape(state_dict[k].shape)
            for k, v in zip(layer_names, decoded_weights)
        }
        self.model.load_state_dict(new_state_dict, strict=True)
        self.logger.info("[DEBUG] Loaded compressed decoded weights.")

    # -----------------------------
    # Fit / Evaluate
    # -----------------------------
    def fit(self, parameters, config):
        torch.cuda.empty_cache()
        self.logger.info("[FIT] Received parameters. Starting local training...")
        rnd = int(config.get("server_round", -1))

        try:
            self.set_parameters(parameters)

            if len(self.train_loader) == 0:
                self.logger.error("[FIT ERROR] Train loader is empty.")
                return self.get_parameters(config, server_round=rnd), 0, {}

            epoch_losses = []
            for epoch in range(self.local_epochs):
                train_loss = train(
                    model=self.model,
                    loader=self.train_loader,
                    epoch=epoch,
                    device=self.device,
                    optimizer=self.optimizer,
                    loss_function=self.loss_function,
                    lr_scheduler=self.lr_scheduler,
                    scaler=self.scaler,
                    logger=self.logger,
                )
                epoch_losses.append(train_loss)

            avg_loss = float(np.mean(epoch_losses))
            self.logger.info(f"[FIT] Local training finished. Avg Loss: {avg_loss:.4f}")

            test_metrics = test(
                model=self.model,
                loader=self.test_loader,
                post_transforms=self.post_transforms,
                device=self.device,
                logger=self.logger,
                dice_metric_batch=self.dice_metric,
            )
            wt_dice = float(test_metrics.get("mean_dice", 0.0))

            # training metrics csv
            if self.metrics_csv_path:
                csv_exists = os.path.exists(self.metrics_csv_path)
                with open(self.metrics_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if not csv_exists:
                        writer.writerow(["Round", "Train Loss", "Dice WT"])
                    writer.writerow([rnd, avg_loss, wt_dice])

            # save local model
            if self.metrics_csv_path:
                model_save_path = os.path.join(os.path.dirname(self.metrics_csv_path), "model_final.pth")
                torch.save(self.model.state_dict(), model_save_path)
                self.logger.info(f"[FIT] Final model saved to: {model_save_path}")

            return self.get_parameters(config, server_round=rnd), len(self.train_loader.dataset), {
                "train_loss": avg_loss,
                "mean_dice": wt_dice,
            }

        except Exception as e:
            self.logger.exception(f"[FIT ERROR] {e}")
            return self.get_parameters(config, server_round=rnd), 0, {}

    def evaluate(self, parameters, config):
        self.logger.info("[EVALUATE] Starting evaluation...")
        try:
            self.set_parameters(parameters)
            metrics = test(
                model=self.model,
                loader=self.test_loader,
                dice_metric_batch=self.dice_metric,
                post_transforms=self.post_transforms,
                device=self.device,
                logger=self.logger,
            )
            wt_dice = float(metrics.get("mean_dice", 0.0))
            self.logger.info(f"[EVALUATE] Dice WT: {wt_dice:.4f}")
            return float(wt_dice), len(self.test_loader.dataset), {"WT": wt_dice}
        except Exception as e:
            self.logger.exception(f"[EVALUATE ERROR] {e}")
            return 0.0, 1, {"WT": 0.0}
