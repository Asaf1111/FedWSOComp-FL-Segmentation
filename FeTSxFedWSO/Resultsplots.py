import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from monai.networks.nets import UNet
from monai.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from clients.BrainTumorSegmentation3dClient.loading_utils import BrainTumorSegmentationCustomDataset
from clients.BrainTumorSegmentation3dClient.utils import val_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)

def visualize_one_slice(
    images, labels, preds, sample_index, dice=None, hd95=None, save_dir=".", prefix="case"
):
    images_np = to_numpy(images)[0]
    labels_np = to_numpy(labels)[0]
    preds_np = to_numpy(preds)[0]

    if labels_np.ndim == 4:
        labels_np = labels_np[0]
    if preds_np.ndim == 4:
        preds_np = preds_np[0]

    bg_index = 3  # T1ce or whichever channel you prefer for visualization
    depth = labels_np.shape[-1]

    # Choose slice with maximum tumor area
    slice_sums = np.sum(labels_np, axis=(0, 1))
    slice_idx = np.argmax(slice_sums) if np.sum(slice_sums) > 0 else depth // 2

    img_slice = images_np[bg_index, :, :, slice_idx]
    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    img_rgb = np.stack([img_norm] * 3, axis=-1)

    gt_mask = labels_np[:, :, slice_idx] > 0
    pred_mask = preds_np[:, :, slice_idx] > 0

    tp = gt_mask & pred_mask
    fn = gt_mask & ~pred_mask
    fp = ~gt_mask & pred_mask

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Frame (grayscale)
    axes[0].imshow(img_rgb, cmap="gray")
    axes[0].set_title("Frame")
    axes[0].axis("off")

    # 2. Ground Truth (binary)
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    # 3. Prediction (binary)
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # 4. Overlap
    axes[3].imshow(img_rgb, cmap="gray")

    # False Negative (GT missed): green
    green = np.zeros((*fn.shape, 4), dtype=np.float32)
    green[..., :3] = [0, 1, 0]
    green[..., 3] = fn * 0.5
    axes[3].imshow(green)

    # False Positive (wrong extra): red
    red = np.zeros((*fp.shape, 4), dtype=np.float32)
    red[..., :3] = [1, 0, 0]
    red[..., 3] = fp * 0.5
    axes[3].imshow(red)

    # True Positive (correct): yellow
    yellow = np.zeros((*tp.shape, 4), dtype=np.float32)
    yellow[..., :3] = [1, 1, 0]
    yellow[..., 3] = tp * 0.5
    axes[3].imshow(yellow)

    axes[3].set_title("Overlap")
    axes[3].axis("off")

    # ---- Legend: centered and aligned under last subfigure (Overlap) ----
    legend_handles = [
        mpatches.Patch(color=(1, 1, 0, 0.5), label="True Positive"),
        mpatches.Patch(color=(0, 1, 0, 0.5), label="False Negative"),
        mpatches.Patch(color=(1, 0, 0, 0.5), label="False Positive"),
    ]

    # Compute horizontal center of the last axes in *figure coordinates*
    ax3_pos = axes[3].get_position()               # Bbox in figure fraction
    ax3_center_x = ax3_pos.x0 + ax3_pos.width / 2  # center of last panel

    # Put legend slightly below the axes row, centered under last subfigure
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(ax3_center_x, 0.02),  # y controls vertical placement
        ncol=1,
        fontsize=10,
        frameon=True,
        facecolor="white",
    )

    # Title with Dice/HD95
    if dice is not None and hd95 is not None:
        sup_title = f"Sample {sample_index} | Dice: {dice:.4f} | HD95: {hd95:.2f}"
    else:
        sup_title = f"Sample {sample_index}"
    fig.suptitle(sup_title, fontsize=12)

    os.makedirs(save_dir, exist_ok=True)

    # Make space at the bottom for the legend
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])

    fig.savefig(
        os.path.join(save_dir, f"{prefix}_sample{sample_index}.pdf"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

def load_model(model_path, device):
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

    state_dict = torch.load(model_path, map_location=device)
    if all(k.startswith("model.") for k in state_dict):
        model.load_state_dict(state_dict)
    else:
        new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.eval()
    return model

def evaluate_model_per_sample(model, loader, device, csv_path, label=""):
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    vis_dir = os.path.join("PredicationPlots/Non-IIDC32S60", label.replace(" ", "_"))
    os.makedirs(vis_dir, exist_ok=True)

    all_scores = []  # (index, dice, hd95, image, label, pred)

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
                all_scores.append((i, dice, hd95, images.cpu(), labels.cpu(), preds.cpu()))

    # Sort by Dice score
    sorted_scores = sorted(all_scores, key=lambda x: x[1])

    # Visualize worst 50
    for rank, (i, dice, hd95, img, lbl, pred) in enumerate(sorted_scores[:50]):
        visualize_one_slice(
            img, lbl, pred,
            sample_index=f"worst{rank}",
            dice=dice,
            hd95=hd95,
            save_dir=vis_dir,
            prefix=label
        )

    # Visualize best 10
    for rank, (i, dice, hd95, img, lbl, pred) in enumerate(sorted_scores[-10:][::-1]):
        visualize_one_slice(
            img, lbl, pred,
            sample_index=f"best{rank}",
            dice=dice,
            hd95=hd95,
            save_dir=vis_dir,
            prefix=label
        )

    avg_dice = np.mean([x[1] for x in all_scores])
    avg_hd95 = np.mean([x[2] for x in all_scores])
    print(f"[{label}] Dice: {avg_dice:.4f} | HD95: {avg_hd95:.4f}")
    return avg_dice, avg_hd95

if __name__ == "__main__":
    MODEL_PATH = "Results/runs/Run_UNET_iid_30%fedWSOcomp_20250526/Server/results/fedWSOcomp_BestRound24.pth"
    DATA_PATH = "/home/jovyan/FeTS2022/MICCAI_FeTS2022_TrainingData"
    SERVER_CSV_PATH = "/home/jovyan/FeTSxFedWSO/ServerTestset.csv"
    RESULT_DIR = "Evaluation_Results_PerSample"
    os.makedirs(RESULT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)

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
