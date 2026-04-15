import torch
import numpy as np
import csv
import os
from monai.networks.nets import UNet
from monai.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from clients.BrainTumorSegmentation3dClient.loading_utils import BrainTumorSegmentationCustomDataset
from clients.BrainTumorSegmentation3dClient.utils import val_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, device):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch',
        dropout=0.2
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)

    if all(k.startswith("model.") for k in state_dict.keys()):
        model.load_state_dict(state_dict)
    else:
        new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.eval()
    return model

def evaluate_model_per_sample(model, loader, device, csv_path, label=""):
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    all_dice, all_hd95 = [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIndex", "Dice", "HD95"])

        with torch.no_grad():
            for i, batch in enumerate(loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                preds = model(images)
                preds = (torch.sigmoid(preds) > 0.5).float()

                dice = dice_metric(preds, labels).item()
                hd95 = hd95_metric(preds, labels)[0].item()

                writer.writerow([i, f"{dice:.4f}", f"{hd95:.4f}"])
                all_dice.append(dice)
                all_hd95.append(hd95)

    avg_dice = np.mean(all_dice)
    avg_hd95 = np.mean(all_hd95)
    print(f"[{label}] Dice: {avg_dice:.4f} | HD95: {avg_hd95:.4f}")
    return avg_dice, avg_hd95

if __name__ == "__main__":
    MODEL_PATH="Results/runs/Run_UNET_iid_fedavg_20250520/Server/results/FederatedModel_BestRound24.pth"
    DATA_PATH = "/home/jovyan/FeTS2022/MICCAI_FeTS2022_TrainingData"
    SERVER_CSV_PATH = "/home/jovyan/FeTSxFedWSO/ServerTestset.csv"
    CLIENT_CSV_DIR = "/home/jovyan/FeTSxFedWSO/data_splitting/clients/iid"
    RESULT_DIR = "Evaluation_Results_PerSample"
    os.makedirs(RESULT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)

    # Server evaluation
    server_ds = BrainTumorSegmentationCustomDataset(
        csv_file=SERVER_CSV_PATH,
        root_dir=DATA_PATH,
        transforms=val_transform,
        device=DEVICE,
        is_server=True
    )
    server_loader = DataLoader(server_ds, batch_size=1, shuffle=False, num_workers=4)
    server_csv_out = os.path.join(RESULT_DIR, "server_per_sample.csv")
    evaluate_model_per_sample(model, server_loader, DEVICE, server_csv_out, label="SERVER")

    # Client evaluations
    for client_id in [1, 2, 3, 4]:
        client_csv_path = os.path.join(CLIENT_CSV_DIR, f"client{client_id}_iid_dataset.csv")
        client_ds = BrainTumorSegmentationCustomDataset(
            csv_file=client_csv_path,
            root_dir=DATA_PATH,
            transforms=val_transform,
            device=DEVICE,
            mode="test",
            val_perc=0.10,
            test_perc=0.10,
            is_server=False
        )
        client_loader = DataLoader(client_ds, batch_size=1, shuffle=False, num_workers=4)
        out_path = os.path.join(RESULT_DIR, f"client{client_id}_per_sample.csv")
        evaluate_model_per_sample(model, client_loader, DEVICE, out_path, label=f"CLIENT {client_id}")