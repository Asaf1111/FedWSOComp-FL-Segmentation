import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import flwr as fl
from datetime import datetime
from torch.amp import GradScaler

import grpc
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import torch
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.data import pad_list_data_collate

from clients.BrainTumorSegmentation3dClient.utils import (
    train_transform, val_transform, post_transforms_with_invert as post_transforms
)
from clients.BrainTumorSegmentation3dClient.loading_utils import BrainTumorSegmentationCustomDataset
from clients.BrainTumorSegmentation3dClient.ClientImpl import BrainTumorSegmentation3dClient

from monai.networks.nets import UNet
from monai.networks.layers import Norm

import threading
import time

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



def ping_forever():
    while True:
        print("[DEBUG] PING still alive")
        time.sleep(300)


threading.Thread(target=ping_forever, daemon=True).start()

# ------------------------ Load .env ------------------------ #
dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path)

ENCODING_MODE = os.getenv("ENCODING_MODE", "SH")  # FEDAVG, SKH, SH, KH
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")
SERVER_PORT = os.getenv("SERVER_PORT")
DATA_PATH = os.getenv("DATA_PATH")
N_CLUSTERS = int(os.getenv("N_CLUSTERS", 32))

EXPERIMENT_DIR, date_str = create_experiment_folder()

assert SERVER_ADDRESS is not None, "SERVER_ADDRESS not found in .env"
assert SERVER_PORT is not None, "SERVER_PORT not found in .env"
assert DATA_PATH is not None, "DATA_PATH not found in .env"

print(f"[DEBUG] Connecting to Flower server at {SERVER_ADDRESS}:{SERVER_PORT}")
print(f"[DEBUG] ENCODING_MODE={ENCODING_MODE}, N_CLUSTERS={N_CLUSTERS}")

# ------------------------ Hyperparameters ------------------------ #
max_local_client_epochs = 5
batch_size_train = 1
batch_size_val = 1
VAL_PERCENTAGE = 0.10
TEST_PERCENTAGE = 0.10
device = "cuda:0"

# ------------------------ Client ID ------------------------ #
client_id = int(sys.argv[1])
client_data_csv=f"/home/jovyan/FeTSxFedWSO/data_splitting/clients/iid/client{client_id}_iid_dataset.csv"
#client_data_csv = f"/home/jovyan/FeTSxFedWSO/data_splitting/clients/non_iid/client{client_id}_noniid_dataset.csv"
# ------------------------ Dataset and Loaders ------------------------ #
train_ds = BrainTumorSegmentationCustomDataset(
    csv_file=client_data_csv,
    root_dir=DATA_PATH,
    transforms=train_transform,
    device=device,
    mode="train",
    val_perc=VAL_PERCENTAGE,
    test_perc=TEST_PERCENTAGE,
    is_server=False,
)

val_ds = BrainTumorSegmentationCustomDataset(
    csv_file=client_data_csv,
    root_dir=DATA_PATH,
    transforms=val_transform,
    device=device,
    mode="val",
    val_perc=VAL_PERCENTAGE,
    test_perc=TEST_PERCENTAGE,
    is_server=False,
)

test_ds = BrainTumorSegmentationCustomDataset(
    csv_file=client_data_csv,
    root_dir=DATA_PATH,
    transforms=val_transform,
    device=device,
    mode="test",
    val_perc=VAL_PERCENTAGE,
    test_perc=TEST_PERCENTAGE,
    is_server=False,
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=pad_list_data_collate,
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size_val,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=pad_list_data_collate,
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size_val,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=pad_list_data_collate,
)

print("Train samples:", len(train_loader.dataset))
print("Val samples:", len(val_loader.dataset))
print("Test samples:", len(test_loader.dataset))

# ------------------------ Model & Optimizer ------------------------ #
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
    dropout=0.2,
).to(device)

loss_function = DiceLoss(
    to_onehot_y=False,
    sigmoid=True,
    include_background=False,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_local_client_epochs
)
scaler = GradScaler()

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

client_dir = os.path.join(EXPERIMENT_DIR, f"Client{client_id}")
os.makedirs(client_dir, exist_ok=True)

log_file_path = os.path.join(client_dir, f"ClientReport{date_str}.log")
CLIENT_METRICS_CSV = os.path.join(client_dir, f"ClientReport{date_str}.csv")

client = BrainTumorSegmentation3dClient(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    scaler=scaler,
    lr_scheduler=lr_scheduler,
    local_epochs=max_local_client_epochs,
    device=device,
    dice_metric=dice_metric,
    post_transforms=post_transforms,

    log_file_path=log_file_path,
    metrics_csv_path=CLIENT_METRICS_CSV,
)



if __name__ == "__main__":
    fl.client.start_client(
        server_address=f"{SERVER_ADDRESS}:{SERVER_PORT}",
        client=client.to_client(),
        grpc_max_message_length=1536870912,
    )
