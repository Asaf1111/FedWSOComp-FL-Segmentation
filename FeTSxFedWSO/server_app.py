# server_app.py

from datetime import datetime
import logging
import os
import time
import csv
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import flwr as fl
from dotenv import load_dotenv
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.data import DataLoader

from clients.BrainTumorSegmentation3dClient.utils import (
    val_transform,
    post_transforms_with_invert as post_transforms,
    test,
)
from clients.BrainTumorSegmentation3dClient.loading_utils import (
    BrainTumorSegmentationCustomDataset,
)
from server_visualize import visualize_server_predictions
from FedWSOCompStrategy import FedWSOCompStrategy, hash_model
from utils import decode_weights


# =========================
# Unified experiment folder
# =========================
def create_experiment_folder(
    model_name="UNET",
    setup="IID",
    extra_tag="SH20",
    base_dir="Revision2",
):
    date_str = datetime.today().strftime("%Y%m%d")
    folder_name = f"Run_{model_name}_{setup}_{extra_tag}_{date_str}"
    full_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path, date_str



# ====== MUST MATCH client_app.py ======
EXPERIMENT_MODEL_NAME = "UNET"
EXPERIMENT_SETUP ="IID"   # e.g. "IID%_SR60", "NonIID%_SR60"
EXPERIMENT_TAG = "SH20"            # FEDAVG / SKH / SH / KH


EXPERIMENT_DIR, date_str = create_experiment_folder(
    model_name=EXPERIMENT_MODEL_NAME,
    setup=EXPERIMENT_SETUP,
    extra_tag=EXPERIMENT_TAG,
)

# -------- Logging setup --------
SERVER_LOG_FILE = os.path.join(
    EXPERIMENT_DIR, "Server", f"ServerReport{date_str}.log"
)
os.makedirs(os.path.dirname(SERVER_LOG_FILE), exist_ok=True)

server_logger = logging.getLogger("server_logger")
server_logger.setLevel(logging.DEBUG)
if not server_logger.hasHandlers():
    sh = logging.FileHandler(SERVER_LOG_FILE)
    sh.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    server_logger.addHandler(sh)

server_logger.info(
    f"[SERVER INIT] EXPERIMENT_DIR={EXPERIMENT_DIR}, LOG={SERVER_LOG_FILE}"
)

# ------------------ Environment Setup ------------------ #
dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")
SERVER_PORT = os.getenv("SERVER_PORT")
DATA_PATH = os.getenv("DATA_PATH")
ENCODING_MODE = os.getenv("ENCODING_MODE", "SKH").upper()  # FEDAVG / SKH / SH / KH
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "32"))

if not SERVER_ADDRESS or not SERVER_PORT:
    raise ValueError("SERVER_ADDRESS or SERVER_PORT is not set. Please check your .env file.")
if not DATA_PATH:
    raise ValueError("DATA_PATH is not set. Please check your .env file.")

server_logger.info(
    f"[ENV] SERVER_ADDRESS={SERVER_ADDRESS}, SERVER_PORT={SERVER_PORT}, "
    f"DATA_PATH={DATA_PATH}, ENCODING_MODE={ENCODING_MODE}, N_CLUSTERS={N_CLUSTERS}"
)

# ------------------ Configuration ------------------ #
CSV_PATH = "/home/jovyan/FeTSxFedWSO/ServerTestset.csv"  # global test CSV
NUM_ROUNDS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = f"UNET_{ENCODING_MODE}"

# ------------------ Load Dataset (Server Test Set) ------------------ #
test_ds = BrainTumorSegmentationCustomDataset(
    csv_file=CSV_PATH,
    root_dir=DATA_PATH,
    transforms=val_transform,
    device=DEVICE,
    is_server=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
    persistent_workers=True,
)

dice_metric_batch = DiceMetric(include_background=False, reduction="mean")

server_logger.info(
    f"[DATA] Server test set loaded with {len(test_loader.dataset)} samples."
)

# ------------------ Initialize model & initial parameters ------------------ #
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
    dropout=0.2,
).to(DEVICE)

state_dict = model.state_dict()
sorted_keys = sorted(state_dict.keys())
initial_weights = [state_dict[k].detach().cpu().numpy() for k in sorted_keys]
initial_parameters = ndarrays_to_parameters(initial_weights)

server_logger.info(
    f"[MODEL INIT] UNet with {len(sorted_keys)} layers initialized on {DEVICE}"
)


# ------------------ Evaluation Function ------------------ #
def get_evaluate_fn(
    model,
    test_loader,
    dice_metric_batch,
    post_transforms,
    device,
    model_name,
    report_path,
    encoding_mode: str,
):
    history = []

    def evaluate(server_round: int, parameters, config):
        server_logger.info(
            f"[SERVER][Round {server_round}] Starting evaluation (mode={encoding_mode})..."
        )
        start_time = time.time()

        ndarrays = parameters_to_ndarrays(parameters)

        try:
            import numpy as _np
            import pickle as _pkl

            # ---------- Detect uncompressed vs compressed ----------
            # Cases:
            #   - Round 0: always uncompressed FP32 from initial_parameters
            #   - FEDAVG: always uncompressed
            #   - SKH/SH/KH (rnd >= 1): compressed as 8 uint8 arrays
            is_probably_compressed = (
                len(ndarrays) == 8
                and isinstance(ndarrays[0], _np.ndarray)
                and ndarrays[0].dtype == _np.uint8
            )

            if encoding_mode.upper() == "FEDAVG" or not is_probably_compressed:
                # Uncompressed path (FEDAVG OR round 0 for SKH/SH/KH)
                decoded_weights = ndarrays
                layer_names = sorted(model.state_dict().keys())
            else:
                # Compressed SKH/SH/KH path (rnd >= 1)
                layer_names = _pkl.loads(ndarrays[7].tobytes())
                encoded_dict = {
                    "centroids": _pkl.loads(ndarrays[0].tobytes()),
                    "encoded_labels": _pkl.loads(ndarrays[1].tobytes()),
                    "label_dicts": _pkl.loads(ndarrays[2].tobytes()),
                    "address_table": _pkl.loads(ndarrays[3].tobytes()),
                    "shapes": _pkl.loads(ndarrays[4].tobytes()),
                    "mode": _pkl.loads(ndarrays[5].tobytes()),
                    "sparse_indices": _pkl.loads(ndarrays[6].tobytes()),
                }
                decoded_weights = decode_weights(encoded_dict)

            # Log decoded weight shapes
            for i, w in enumerate(decoded_weights):
                print(
                    f"[DEBUG][SERVER-EVAL] Decoded layer {i}: "
                    f"shape={w.shape if hasattr(w, 'shape') else 'N/A'}"
                )

            # Load into model
            new_state_dict = OrderedDict()
            state_dict_local = model.state_dict()

            for idx, (k, v) in enumerate(zip(layer_names, decoded_weights)):
                expected_shape = state_dict_local[k].shape
                try:
                    reshaped_tensor = (
                        torch.tensor(v, dtype=torch.float32, device=device)
                        .reshape(expected_shape)
                    )
                    new_state_dict[k] = reshaped_tensor
                except Exception as e:
                    print(f"[✖ CRITICAL SHAPE ERROR] Layer {idx} = {k}")
                    print(f"    Expected shape: {expected_shape}")
                    print(
                        f"    Got array shape: "
                        f"{v.shape if hasattr(v, 'shape') else 'non-numpy'}"
                    )
                    print(
                        f"    Raw array length: {len(v) if hasattr(v, '__len__') else 'N/A'}"
                    )
                    raise e

            model.load_state_dict(new_state_dict, strict=True)
            print("[SERVER DEBUG] Loaded model summary:")
            for name, param in model.named_parameters():
                print(
                    f"   {name} — mean: {param.data.mean():.4f}, "
                    f"std: {param.data.std():.4f}, "
                    f"non-zeros: {torch.count_nonzero(param.data)}"
                )
            print(f"[DEBUG] Evaluated model hash: {hash_model(model)}")

        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load decoded weights to model: {e}")

        # ---------------- Evaluate on server test set ---------------- #
        model.eval()

        server_dir = os.path.join(report_path, "Server")
        os.makedirs(server_dir, exist_ok=True)

        csv_path = os.path.join(
            server_dir, f"PerSampleDice_Round{server_round}.csv"
        )
        metrics = test(
            model=model,
            loader=test_loader,
            dice_metric_batch=dice_metric_batch,
            post_transforms=post_transforms,
            device=device,
            save_csv=True,
            csv_path=csv_path,
        )

        mean_dice = float(metrics["mean_dice"])
        duration = time.time() - start_time
        server_logger.info(
            f"[Round {server_round}] WT Dice: {mean_dice:.4f} | Time: {duration:.2f}s"
        )

        # ---------------- Write ServerReport.csv ---------------- #
        report_csv = os.path.join(server_dir, "ServerReport.csv")
        fields = ["round", "dice_wt", "time_sec", "loss"]
        loss = 1.0 - mean_dice
        row = [server_round, mean_dice, round(duration, 2), round(loss, 4)]

        try:
            write_header = (
                not os.path.exists(report_csv) or os.stat(report_csv).st_size == 0
            )
            with open(report_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(fields)
                writer.writerow(row)
        except Exception as e:
            server_logger.error(f"[ServerReport CSV ERROR] {e}")

        # ---------------- Visualize final round predictions ---------------- #
        if server_round == NUM_ROUNDS:
            vis_dir = os.path.join(server_dir, "results", "test_predictions")
            os.makedirs(vis_dir, exist_ok=True)
            visualize_server_predictions(
                model=model,
                loader=test_loader,
                save_dir=vis_dir,
                device=device,
            )

        # Track best Dice in memory
        history.append(mean_dice)

        return 1.0 - mean_dice, {
            "WT": mean_dice,
            "num_examples": len(test_loader.dataset),
        }

    return evaluate


# ------------------ Create strategy and start server ------------------ #
strategy = FedWSOCompStrategy(
    min_fit_clients=4,
    min_available_clients=4,
    min_evaluate_clients=4,
    mode=ENCODING_MODE,          # FEDAVG / SKH / SH / KH
    n_clusters=N_CLUSTERS,
    initial_parameters=initial_parameters,
    experiment_dir=EXPERIMENT_DIR,
    evaluate_fn=None,
)

strategy.evaluate_fn = get_evaluate_fn(
    model=strategy.get_model(),
    test_loader=test_loader,
    dice_metric_batch=dice_metric_batch,
    device=DEVICE,
    report_path=EXPERIMENT_DIR,
    model_name=MODEL_NAME,
    post_transforms=post_transforms,
    encoding_mode=ENCODING_MODE,
)

server_logger.info(
    f"[SERVER START] Address={SERVER_ADDRESS}:{SERVER_PORT}, "
    f"Rounds={NUM_ROUNDS}, Mode={ENCODING_MODE}, NClusters={N_CLUSTERS}"
)

fl.server.start_server(
    server_address=f"{SERVER_ADDRESS}:{SERVER_PORT}",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy,
)
