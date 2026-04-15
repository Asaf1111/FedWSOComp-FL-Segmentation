import torch
import numpy as np
import csv
import os
from monai.networks.nets import UNet
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import Compose, Activationsd, AsDiscreted

from clients.BrainTumorSegmentation3dClient.loading_utils import BrainTumorSegmentationCustomDatasetExtended
from clients.BrainTumorSegmentation3dClient.utils import val_transform, ConvertToWholeTumorOnlyd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os

DATA_PATH = "/home/jovyan/FeTS2022/MICCAI_FeTS2022_TrainingData"
SERVER_CSV_PATH = "/home/jovyan/FeTSxFedWSO/ServerTestset.csv"
CLIENT_CSV_DIR = "/home/jovyan/FeTSxFedWSO/data_splitting/clients/iid"

print(f"Server CSV exists: {os.path.exists(SERVER_CSV_PATH)}")
print(f"Client CSV directory exists: {os.path.exists(CLIENT_CSV_DIR)}")

if os.path.exists(CLIENT_CSV_DIR):
    print("\nFiles in client directory:")
    for file in sorted(os.listdir(CLIENT_CSV_DIR)):
        print(f"  {file}")
    
    # Check for client1 files specifically
    print("\nLooking for client1 files:")
    for file in sorted(os.listdir(CLIENT_CSV_DIR)):
        if 'client1' in file.lower():
            print(f"  Found: {file}")

post_transforms = Compose([
    Activationsd(keys="pred", sigmoid=True),
    AsDiscreted(keys="pred", threshold=0.5),
    ConvertToWholeTumorOnlyd(keys=["label"]),
    AsDiscreted(keys="label", threshold=0.5),
])

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
        model.load_state_dict({f"model.{k}": v for k, v in state_dict.items()})
    model.eval()
    return model

def evaluate_model_per_sample(model, loader, device, csv_path, label=""):
    dice_scores, hd95_scores = [], []
    hausdorff = HausdorffDistanceMetric(include_background=False, percentile=95)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "Dice", "HD95"])

        with torch.no_grad():
            for i, (batch, sample_id) in enumerate(loader):
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = sliding_window_inference(
                    inputs, roi_size=inputs.shape[2:], sw_batch_size=1, predictor=model
                )

                outputs = decollate_batch(outputs)
                labels = decollate_batch(labels)
                samples = [post_transforms({"pred": p, "label": l}) for p, l in zip(outputs, labels)]

                for j, sample in enumerate(samples):
                    pred = sample["pred"]
                    label_t = sample["label"]

                    intersection = torch.sum((pred == 1) & (label_t == 1)).item()
                    union = torch.sum((pred == 1) | (label_t == 1)).item()
                    dice = 1.0 if union == 0 and intersection == 0 else 2 * intersection / (union + 1e-8)
                    dice = float(min(max(dice, 0.0), 1.0))

                    hd = hausdorff(pred.unsqueeze(0), label_t.unsqueeze(0)).item()
                    writer.writerow([f"{i}_{j}", f"{dice:.4f}", f"{hd:.4f}"])
                    dice_scores.append(dice)
                    hd95_scores.append(hd)

        # Append aggregated results
        avg_dice = np.mean(dice_scores)
        avg_hd95 = np.mean(hd95_scores)
        writer.writerow(["Average", f"{avg_dice:.4f}", f"{avg_hd95:.4f}"])

    print(f"[{label}] Mean Dice: {avg_dice:.4f} | Mean HD95: {avg_hd95:.4f}")
    return avg_dice, avg_hd95

if __name__ == "__main__":
    MODEL_PATH="Revision2/Run_UNET_NonIID_SH20_20260129/Server/best_model.pth"
    DATA_PATH = "/home/jovyan/FeTS2022/MICCAI_FeTS2022_TrainingData"
    SERVER_CSV_PATH = "/home/jovyan/FeTSxFedWSO/ServerTestset.csv"
    CLIENT_CSV_DIR = "/home/jovyan/FeTSxFedWSO/data_splitting/clients/non_iid"
    RESULT_DIR = "Results/IPTestR/SH20_NonIID"
    SUMMARY_CSV = os.path.join(RESULT_DIR, "summary.csv")

    os.makedirs(RESULT_DIR, exist_ok=True)
    model = load_model(MODEL_PATH, DEVICE)

    # Collect results
    summary_results = []

    # --- Server evaluation ---
    server_ds = BrainTumorSegmentationCustomDatasetExtended(
        csv_file=SERVER_CSV_PATH,
        root_dir=DATA_PATH,
        transforms=val_transform,
        device=DEVICE,
        is_server=True
    )
    server_loader = DataLoader(server_ds, batch_size=1, shuffle=False, num_workers=4)
    server_csv = os.path.join(RESULT_DIR, "server_per_sample.csv")
    dice, hd95 = evaluate_model_per_sample(model, server_loader, DEVICE, server_csv, label="SERVER")
    summary_results.append(["Server", f"{dice:.4f}", f"{hd95:.4f}"])

    # --- Client evaluations ---
    for client_id in [1, 2, 3, 4]:
        client_csv = os.path.join(CLIENT_CSV_DIR, f"client{client_id}_noniid_dataset.csv")
        client_ds = BrainTumorSegmentationCustomDatasetExtended(
            csv_file=client_csv,
            root_dir=DATA_PATH,
            transforms=val_transform,
            device=DEVICE,
            mode="test",
            val_perc=0.10,
            test_perc=0.10,
            is_server=False
        )
        client_loader = DataLoader(client_ds, batch_size=1, shuffle=False, num_workers=4)
        client_csv_out = os.path.join(RESULT_DIR, f"client{client_id}_per_sample.csv")
        dice, hd95 = evaluate_model_per_sample(model, client_loader, DEVICE, client_csv_out, label=f"CLIENT {client_id}")
        summary_results.append([f"Client {client_id}", f"{dice:.4f}", f"{hd95:.4f}"])

    # Save aggregated summary
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity", "Mean Dice", "Mean HD95"])
        writer.writerows(summary_results)

    print("\n=== Aggregated Summary ===")
    for row in summary_results:
        print(f"{row[0]} -> Dice: {row[1]} | HD95: {row[2]}")


