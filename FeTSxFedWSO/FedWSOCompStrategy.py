# FedWSOCompStrategy.py
# FULL FILE (UPDATED):
# ✅ Adds BEST MODEL saving in Server folder:
#   - Saves best global model as:
#       <experiment_dir>/Server/best_model.pth   (overwritten when improved)
#       <experiment_dir>/Server/best_model_round{r}.pth (snapshot)
#   - Logs best tracking to:
#       <experiment_dir>/Server/BestModelInfo.csv
# ✅ Works for BOTH:
#   - FEDAVG (raw float weights)
#   - SKH/SH/KH (uint8 pickled payload -> decode_weights)
#
# Keeps your previous fixes:
# - int64 aggregation safety (num_batches_tracked etc.)
# - FEDAVG path uses raw FedAvg aggregation (no decode/encode)
# - compressed path decode->aggregate->encode
# - ServerCommReport.csv UsedClusters logic
# - env sparsity usage
# - safer n_clusters default=32
# - init log shows UsedClusters vs EnvNClusters

import os
import csv
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import flwr as fl
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from utils import decode_weights, encode_weights, estimate_comm_bits_theoretical


# -------------------------
# Helpers
# -------------------------
def _is_integer_array(x: np.ndarray) -> bool:
    """True for int/bool arrays that must NOT be averaged."""
    return np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_)


def _weighted_average_ndarrays(
    weights_results: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """
    Safely aggregate a list of (ndarray_list, num_examples).
    - float tensors: weighted mean in float32
    - int/bool tensors: copy from first client (do not average)
    """
    if not weights_results:
        raise ValueError("weights_results is empty")

    total_examples = float(sum(n for _, n in weights_results))
    if total_examples <= 0:
        raise ValueError("total_examples must be > 0")

    num_layers = len(weights_results[0][0])
    aggregated: List[np.ndarray] = []

    for i in range(num_layers):
        ref = weights_results[0][0][i]

        # int/bool params (e.g., BatchNorm num_batches_tracked): do NOT average
        if _is_integer_array(ref):
            aggregated.append(ref.copy())
            continue

        acc = np.zeros(ref.shape, dtype=np.float32)
        for w_list, n in weights_results:
            acc += w_list[i].astype(np.float32) * float(n)
        acc /= total_examples
        aggregated.append(acc)

    return aggregated


def log_and_plot_sparsity(stage1_weights, round_num, experiment_dir, logger):
    total_nonzero = 0
    total_params = 0

    for i, w in enumerate(stage1_weights):
        nonzero = np.count_nonzero(w)
        total_nonzero += nonzero
        total_params += w.size
        logger.debug(f"[SPARSITY] Layer {i}: non-zeros = {nonzero} / {w.size}")

    global_sparsity = (1 - total_nonzero / total_params) * 100
    logger.info(f"[SPARSITY] Global sparsity at round {round_num}: {global_sparsity:.2f}%")

    nonzeros = [np.count_nonzero(w) for w in stage1_weights]
    plt.figure(figsize=(8, 6))
    plt.hist(nonzeros, bins=20)
    plt.title("Layer-wise Non-Zero Weights After Sparsification")
    plt.xlabel("Non-Zero Count per Layer")
    plt.ylabel("Frequency")
    plt.grid(True)

    plot_path = os.path.join(experiment_dir, "Server", f"NonZeroHistogram_Round{round_num}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return global_sparsity, plot_path


class FedWSOCompStrategy(fl.server.strategy.FedAvg):
    """
    Supports:
      - FEDAVG: no compression
      - SKH: Sparsification + KMeans + Huffman
      - SH : Sparsification + Huffman (NO KMeans)  [true SH requires utils.py mode support]
      - KH : KMeans + Huffman (NO Sparsification) [true KH requires utils.py mode support]
    """

    def __init__(
        self,
        evaluate_fn=None,
        mode: str = "SH",
        n_clusters: int = 32,  # ✅ safer default
        experiment_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.experiment_dir = experiment_dir
        self.encoding_mode = (mode or "SKH").upper()
        self.n_clusters = int(n_clusters)
        self.logger = logging.getLogger("server_logger")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.2,
        ).to(self.device)

        # ---- Best model tracking ----
        # Track best by Dice (WT) by default
        self.best_metric_key = os.getenv("BEST_METRIC_KEY", "WT")  # WT / mean_dice etc.
        self.best_metric_value = float("-inf")
        self.best_round = -1

        mode_u = self.encoding_mode
        if mode_u == "FEDAVG":
            used_clusters = "0"
        elif mode_u == "SH":
            used_clusters = "-"
        else:  # SKH / KH
            used_clusters = str(self.n_clusters)

        self.logger.info(
            f"[STRATEGY INIT] mode={mode_u}, UsedClusters={used_clusters} "
            f"(EnvNClusters={self.n_clusters}), experiment_dir={self.experiment_dir} "
            f"| BestMetricKey={self.best_metric_key}"
        )

    def get_model(self):
        return self.model

    # -------------------------
    # Model param setter
    # -------------------------
    def set_model_params(self, weights: List[np.ndarray], layer_names: List[str]) -> None:
        state_dict = self.model.state_dict()
        self.logger.info(f"[DEBUG][SERVER] Model expects {len(state_dict)} layers")

        new_state_dict = {}
        for k, w in zip(layer_names, weights):
            expected_tensor = state_dict[k]
            try:
                decoded_np = np.array(w)

                # Ensure dtype matches: floats -> float32; ints/bools -> int64/bool
                if expected_tensor.dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
                    if decoded_np.dtype not in (np.int64, np.bool_):
                        decoded_np = decoded_np.astype(np.int64, copy=False)
                else:
                    if decoded_np.dtype != np.float32:
                        decoded_np = decoded_np.astype(np.float32, copy=False)

                # Handle scalar shapes
                if expected_tensor.shape == torch.Size([]) and decoded_np.shape in [(), (1,)]:
                    decoded_np = decoded_np.reshape(())
                elif expected_tensor.shape == torch.Size([1]) and decoded_np.shape in [(), (1,)]:
                    decoded_np = decoded_np.reshape((1,))

                decoded = torch.tensor(decoded_np, device=self.device).reshape(expected_tensor.shape)
                new_state_dict[k] = decoded

            except Exception as e:
                self.logger.exception(f"[CRITICAL] Failed to reshape/load layer {k}: {e}")
                new_state_dict[k] = torch.zeros_like(expected_tensor)

        self.model.load_state_dict(new_state_dict, strict=True)
        self.logger.info(f"[DEBUG] Loaded {len(new_state_dict)} of {len(state_dict)} layers")

    # -------------------------
    # Decode pickled payload (same format as client/server broadcast for compressed modes)
    # -------------------------
    def _decode_pickled_payload_to_weights(
        self, parameters: fl.common.Parameters
    ) -> Tuple[List[np.ndarray], List[str]]:
        ndarrays = parameters_to_ndarrays(parameters)

        decoded_np = []
        for arr in ndarrays:
            try:
                obj = pickle.loads(arr.tobytes())
            except Exception:
                obj = arr.tobytes().decode("utf-8", errors="ignore")
            decoded_np.append(obj)

        encoded_dict = {
            "centroids": decoded_np[0],
            "encoded_labels": decoded_np[1],
            "label_dicts": decoded_np[2],
            "address_table": decoded_np[3],
            "shapes": [tuple(int(round(s)) for s in shape) for shape in decoded_np[4]],
            "mode": decoded_np[5],
            "sparse_indices": decoded_np[6],
        }
        layer_names = decoded_np[7]
        decoded_weights = decode_weights(encoded_dict)
        return decoded_weights, layer_names

    # -------------------------
    # Decode client payload (compressed)
    # -------------------------
    def decode_client_weights(
        self, parameters: fl.common.Parameters
    ) -> Tuple[List[np.ndarray], List[str]]:
        try:
            return self._decode_pickled_payload_to_weights(parameters)
        except Exception as e:
            self.logger.error("[ERROR] decode_client_weights failed", exc_info=True)
            raise RuntimeError(f"decode_client_weights error: {e}")

    # -------------------------
    # BEST MODEL saving helpers
    # -------------------------
    def _server_dir(self) -> Optional[str]:
        if self.experiment_dir is None:
            return None
        d = os.path.join(self.experiment_dir, "Server")
        os.makedirs(d, exist_ok=True)
        return d

    def _write_best_info_csv(self, server_round: int, loss: Optional[float], metrics: Dict[str, Any]):
        server_dir = self._server_dir()
        if server_dir is None:
            return
        csv_path = os.path.join(server_dir, "BestModelInfo.csv")
        exists = os.path.exists(csv_path)
        try:
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["Round", "BestMetricKey", "MetricValue", "Loss", "AllMetrics"])
                w.writerow([
                    int(server_round),
                    str(self.best_metric_key),
                    float(self.best_metric_value),
                    "" if loss is None else float(loss),
                    dict(metrics),
                ])
        except Exception as e:
            self.logger.error(f"[BestModelInfo CSV ERROR] {e}")

    def _extract_metric_value(self, metrics: Dict[str, Any]) -> Optional[float]:
        """
        Prefer BEST_METRIC_KEY (default WT). Fallbacks: mean_dice, WT, first numeric metric.
        """
        if not metrics:
            return None

        # prefer configured key
        if self.best_metric_key in metrics:
            try:
                return float(metrics[self.best_metric_key])
            except Exception:
                pass

        # common fallbacks
        for k in ["WT", "mean_dice", "dice_wt", "Dice", "dice"]:
            if k in metrics:
                try:
                    return float(metrics[k])
                except Exception:
                    pass

        # any numeric metric
        for _, v in metrics.items():
            try:
                fv = float(v)
                return fv
            except Exception:
                continue
        return None

    def _save_model_state_dict(self, path: str):
        try:
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"[BEST MODEL] Saved: {path}")
        except Exception as e:
            self.logger.error(f"[BEST MODEL SAVE ERROR] {e}", exc_info=True)

    def _maybe_update_best(self, server_round: int, loss: Optional[float], metrics: Dict[str, Any]):
        val = self._extract_metric_value(metrics)
        if val is None:
            return

        if val > self.best_metric_value:
            self.best_metric_value = float(val)
            self.best_round = int(server_round)

            server_dir = self._server_dir()
            if server_dir is not None:
                # overwrite best + snapshot
                self._save_model_state_dict(os.path.join(server_dir, "best_model.pth"))
                self._save_model_state_dict(os.path.join(server_dir, f"best_model_round{server_round}.pth"))

            self.logger.info(
                f"[BEST MODEL] New best at round {server_round}: {self.best_metric_key}={self.best_metric_value:.4f}"
            )
            self._write_best_info_csv(server_round, loss, metrics)

    # -------------------------
    # Aggregate Fit (core)
    # -------------------------
    def aggregate_fit(self, rnd, results, failures):
        self.logger.info(f"[SERVER DEBUG] aggregate_fit called at round {rnd}")

        if failures:
            self.logger.warning(f"[Round {rnd}] {len(failures)} client failures")
            return None, {}

        if not results:
            self.logger.error("[Server] No client results in aggregate_fit.")
            return None, {}

        self.logger.info(f"[DEBUG] aggregate_fit received {len(results)} results at round {rnd}")

        # =========================================================
        # ✅ FEDAVG PATH (NO DECODE, NO ENCODE)
        # =========================================================
        if self.encoding_mode == "FEDAVG":
            weights_results: List[Tuple[List[np.ndarray], int]] = []
            for _, res in results:
                nds = parameters_to_ndarrays(res.parameters)  # raw weights
                weights_results.append((nds, res.num_examples))

            aggregated = _weighted_average_ndarrays(weights_results)
            layer_names = sorted(self.model.state_dict().keys())

            # Round 0: initialize + broadcast uncompressed
            if rnd == 0:
                self.set_model_params(aggregated, layer_names)
                return ndarrays_to_parameters(aggregated), {}

            baseline_bits = float(sum(int(w.size) * 32 for w in aggregated))
            compressed_bits = baseline_bits
            used_clusters = "0"

            self._log_server_comm_csv(
                rnd=rnd,
                mode="FEDAVG",
                used_clusters=used_clusters,
                baseline_bits=baseline_bits,
                compressed_bits=compressed_bits,
            )

            self.set_model_params(aggregated, layer_names)
            return ndarrays_to_parameters(aggregated), {}

        # =========================================================
        # ✅ COMPRESSED PATH (SKH / SH / KH)
        # Decode -> weighted average -> encode -> broadcast
        # =========================================================
        decoded_weights_list: List[Tuple[List[np.ndarray], int]] = []
        layer_names = None

        for cid, res in results:
            self.logger.info(f"[DEBUG] Decoding client response from {cid.cid}...")
            try:
                decoded, layers = self.decode_client_weights(res.parameters)
                decoded_weights_list.append((decoded, res.num_examples))
                if layer_names is None:
                    layer_names = layers
            except Exception as e:
                self.logger.error(f"[Server] Failed to decode client {cid.cid}: {e}", exc_info=True)

        if not decoded_weights_list:
            self.logger.error("[Server] No valid updates received.")
            return None, {}

        aggregated = _weighted_average_ndarrays(decoded_weights_list)

        # Round 0: send uncompressed baseline (keeps your earlier behavior)
        if rnd == 0:
            self.set_model_params(aggregated, layer_names)
            self.logger.info(f"[Round {rnd}] Aggregated global model updated (uncompressed)")
            return ndarrays_to_parameters(aggregated), {}

        baseline_bits = float(sum(int(w.size) * 32 for w in aggregated))

        # Sparsity cfg from env
        sparsity_cfg = {
            "Conv": float(os.getenv("SPARSITY_CONV", 0.60)),
            "Linear": float(os.getenv("SPARSITY_LINEAR", 0.60)),
            "Bias": float(os.getenv("SPARSITY_BIAS", 0.0)),
            "BatchNorm": float(os.getenv("SPARSITY_BN", 0.0)),
        }

        # Encode aggregated weights
        self.logger.info("[ENCODE] Starting encode_weights in aggregate_fit...")
        encoded_result = encode_weights(
            aggregated,
            sparsity_cfg=sparsity_cfg,
            n_clusters=self.n_clusters,
        )
        self.logger.info("[ENCODE] Finished encode_weights.")

        compressed_bits = float(estimate_comm_bits_theoretical(encoded_result, sparsity_cfg))
        used_clusters = "-" if self.encoding_mode == "SH" else str(self.n_clusters)

        self._log_server_comm_csv(
            rnd=rnd,
            mode=self.encoding_mode,
            used_clusters=used_clusters,
            baseline_bits=baseline_bits,
            compressed_bits=compressed_bits,
        )

        # Update server-side model with aggregated decoded weights (for evaluation)
        self.set_model_params(aggregated, layer_names)

        # Broadcast encoded payload
        def serialize(obj):
            return np.frombuffer(pickle.dumps(obj), dtype=np.uint8)

        param_list = [
            serialize(encoded_result["centroids"]),
            serialize(encoded_result["encoded_labels"]),
            serialize(encoded_result["label_dicts"]),
            serialize(encoded_result["address_table"]),
            serialize(encoded_result["shapes"]),
            serialize(encoded_result["mode"]),
            serialize(encoded_result["sparse_indices"]),
            serialize(layer_names),
        ]
        return ndarrays_to_parameters(param_list), {}

    # -------------------------
    # Evaluate hook (UPDATED: best model saving)
    # -------------------------
    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
        if self.evaluate_fn is None:
            return None

        # Ensure server model matches the parameters being evaluated, so saved model is consistent
        try:
            nds = parameters_to_ndarrays(parameters)
            if len(nds) == 0:
                raise ValueError("Empty parameters")

            # Case A: raw weights (FedAvg OR round0 baseline)
            if nds[0].dtype != np.uint8:
                layer_names = sorted(self.model.state_dict().keys())
                self.set_model_params(nds, layer_names)

            # Case B: encoded payload (uint8 pickles)
            else:
                decoded_weights, layer_names = self._decode_pickled_payload_to_weights(parameters)
                self.set_model_params(decoded_weights, layer_names)

        except Exception as e:
            self.logger.error(f"[EVAL SYNC ERROR] Could not sync model params before eval: {e}", exc_info=True)

        self.logger.info(f"[Server] Evaluating at round {server_round}")
        out = self.evaluate_fn(server_round, parameters, {})

        # Flower evaluate_fn typically returns: (loss, metrics_dict)
        loss = None
        metrics: Dict[str, Any] = {}
        try:
            if isinstance(out, tuple) and len(out) == 2:
                loss = float(out[0]) if out[0] is not None else None
                metrics = out[1] if isinstance(out[1], dict) else {}
            else:
                # unexpected shape, keep safe
                metrics = {}
        except Exception:
            metrics = {}

        # Update best model (based on BEST_METRIC_KEY)
        try:
            self._maybe_update_best(server_round, loss, metrics)
        except Exception as e:
            self.logger.error(f"[BEST MODEL TRACKING ERROR] {e}", exc_info=True)

        return out

    # -------------------------
    # Comm CSV logger
    # -------------------------
    def _log_server_comm_csv(
        self,
        rnd: int,
        mode: str,
        used_clusters: str,
        baseline_bits: float,
        compressed_bits: float,
    ):
        if self.experiment_dir is None:
            return

        server_dir = os.path.join(self.experiment_dir, "Server")
        os.makedirs(server_dir, exist_ok=True)
        csv_path = os.path.join(server_dir, "ServerCommReport.csv")

        baseline_bits = float(baseline_bits)
        compressed_bits = float(compressed_bits)
        cr = (baseline_bits / compressed_bits) if compressed_bits > 0 else 1.0

        csv_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    writer.writerow(
                        [
                            "Round",
                            "EncodingMode",
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
                        str(used_clusters),  # "-" allowed for SH
                        int(baseline_bits),
                        int(compressed_bits),
                        float(cr),
                    ]
                )
        except Exception as e:
            self.logger.error(f"[ServerCommReport CSV ERROR] {e}")


def hash_model(model):
    import hashlib
    total = b"".join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    return hashlib.md5(total).hexdigest()
