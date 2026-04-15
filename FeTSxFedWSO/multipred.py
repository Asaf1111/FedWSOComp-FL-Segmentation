import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple

from monai.networks.nets import UNet
from monai.data import DataLoader

from clients.BrainTumorSegmentation3dClient.loading_utils import BrainTumorSegmentationCustomDataset
from clients.BrainTumorSegmentation3dClient.utils import val_transform

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BG_CHANNEL_INDEX = 3
THRESH = 0.5
FIG_DPI = 300

# Colors
COLOR_GT = np.array([0.00, 0.60, 0.00], dtype=np.float32)  # Ground truth = green (opaque)
COLOR_PR = np.array([0.70, 0.00, 0.00], dtype=np.float32)  # Prediction  = red (semi-transparent)

# Transparency
ALPHA_PR = 0.45      # PNG / on-screen
PDF_ALPHA_PR = 0.60  # a bit stronger for PDF precomposition

NUM_WORKERS = 8
K_PER_CATEGORY = 10
RESULT_DIR = "Plots/FedWSOComp_Comparative_Matrix"
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# Utilities
# =========================
def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def load_model(model_path: str, device: str) -> UNet:
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm="batch", dropout=0.2
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if all(k.startswith("model.") for k in state_dict):
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict({f"model.{k}": v for k, v in state_dict.items()})
    model.eval()
    return model

def predict_bin(model: UNet, volume: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(volume)
        pred = (torch.sigmoid(logits) > THRESH).float()
    return pred

def dice_fg(pred_bin: torch.Tensor, lbl: torch.Tensor) -> float:
    p = (pred_bin > 0).float(); g = (lbl > 0).float()
    inter = (p * g).sum().item()
    p_sum, g_sum = p.sum().item(), g.sum().item()
    if p_sum == 0 and g_sum == 0: return 1.0
    return float((2.0 * inter) / (p_sum + g_sum + 1e-8))

def slice_with_max_disagreement(gt3d: np.ndarray, pred3d: np.ndarray) -> int:
    gt = gt3d.astype(bool); pr = pred3d.astype(bool)
    D = gt.shape[-1]; scores = []
    for d in range(D):
        fn = np.logical_and(gt[..., d], np.logical_not(pr[..., d])).sum()
        fp = np.logical_and(np.logical_not(gt[..., d]), pr[..., d]).sum()
        scores.append(int(fn + fp))
    best = int(np.argmax(scores))
    if scores[best] == 0:
        gt_sums = [int(gt[..., d].sum()) for d in range(D)]
        best = int(np.argmax(gt_sums)) if np.max(gt_sums) > 0 else D // 2
    return best

def score_all_samples_by_dice(dataset, ref_model_path: str) -> List[Tuple[int, float]]:
    model = load_model(ref_model_path, DEVICE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE == "cuda"),
                        persistent_workers=(NUM_WORKERS > 0))
    scores = []
    for i, batch in enumerate(loader):
        img = batch["image"].to(DEVICE).float()
        lbl = batch["label"].to(DEVICE).float()
        pred_bin = predict_bin(model, img)
        d = dice_fg(pred_bin, lbl)
        scores.append((i, d))
    return scores

def select_k_worst_median_best(scores: List[Tuple[int, float]], k: int):
    scores_sorted = sorted(scores, key=lambda x: x[1])
    n = len(scores_sorted); k = min(k, n)
    worst = [scores_sorted[i][0] for i in range(k)]
    best  = [scores_sorted[-(i+1)][0] for i in range(k)]
    mid = n // 2; half = k // 2
    start = max(0, min(n - k, mid - half))
    median = [scores_sorted[start + i][0] for i in range(k)]
    return worst, median, best

# =========================
# Parse labels -> regime, C, S
# =========================
LABEL_RE = re.compile(r"^(IID|NonIID)\s+C(\d+)\s+S(\d+)$")
def parse_label(label: str):
    m = LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"Label not in expected format: {label}")
    regime, c_str, s_str = m.group(1), m.group(2), m.group(3)
    return regime, int(c_str), int(s_str)

# =========================
# Compositing (GT opaque, Pred semi-transparent)
# =========================
def compose_gt_pred(frame_rgb: np.ndarray,
                    gt: np.ndarray, pred: np.ndarray,
                    alpha_pred: float) -> np.ndarray:
    out = frame_rgb.copy().astype(np.float32)
    a_gt = gt[..., None].astype(np.float32)
    out = out * (1 - a_gt) + COLOR_GT.reshape(1, 1, 3) * a_gt
    a_pr = (pred[..., None].astype(np.float32) * alpha_pred)
    out = out * (1 - a_pr) + COLOR_PR.reshape(1, 1, 3) * a_pr
    return np.clip(out, 0, 1)

# =========================
# Matrix Figure (robust suptitle centering)
# =========================
def make_matrix_figure(configs: List[Dict], dataset, sample_index: int,
                       ref_model_path: str, for_pdf: bool, regime_title: str):
    """
    Rows: S = 20, 30, 60
    Cols: Ground truth | C=3 | C=32 | C=64
    Legend: bottom-right; Title: bold and truly centered over *all* content (plots + row labels).
    """
    # Sample + slice selection
    sample = dataset[sample_index]
    img = sample["image"].unsqueeze(0).to(DEVICE).float()
    lbl = sample["label"].unsqueeze(0).to(DEVICE).float()

    ref_model = load_model(ref_model_path, DEVICE)
    ref_pred_bin = predict_bin(ref_model, img)
    gt3d = to_numpy(lbl > 0)[0, 0].astype(bool)
    pred3d = to_numpy(ref_pred_bin > 0)[0, 0].astype(bool)
    slice_idx = slice_with_max_disagreement(gt3d, pred3d)

    # Slice data
    img_np = to_numpy(img)[0]
    img_slice = img_np[BG_CHANNEL_INDEX, :, :, slice_idx]
    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    frame_rgb = np.stack([img_norm]*3, axis=-1)
    gt2d = gt3d[..., slice_idx]

    # Map (S,C) -> path
    model_map: Dict[Tuple[int, int], str] = {}
    for m in configs:
        _, C, S = parse_label(m["label"])
        model_map[(S, C)] = m["path"]

    S_vals = [20, 30, 60]
    C_vals = [3, 32, 64]

    # Predictions
    cell = {}
    with torch.no_grad():
        for S in S_vals:
            for C in C_vals:
                path = model_map.get((S, C), None)
                if path is None or not os.path.exists(path):
                    cell[(S, C)] = None
                    continue
                model = load_model(path, DEVICE)
                pred_bin = predict_bin(model, img)
                pred_np = to_numpy(pred_bin > 0)[0, 0].astype(bool)
                cell[(S, C)] = {"pred": pred_np[..., slice_idx]}

    # Grid
    nrows = len(S_vals)
    ncols = 1 + len(C_vals)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), dpi=FIG_DPI)
    if nrows == 1:
        axes = np.array([axes])

    # Layout margins (extra left for row labels)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)

    # Alpha
    a_pr = PDF_ALPHA_PR if for_pdf else ALPHA_PR

    # Draw panels + collect row label Text artists
    row_label_texts = []
    for i, S in enumerate(S_vals):
        for j in range(ncols):
            ax = axes[i, j]
            ax.set_aspect("equal"); ax.axis("off")

            if j == 0:
                ax.imshow(compose_gt_pred(frame_rgb, gt2d, np.zeros_like(gt2d), 0.0))
                if i == 0:
                    ax.set_title("Ground truth", fontweight="bold")
                continue

            C = C_vals[j - 1]
            entry = cell.get((S, C), None)
            if entry is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9, color="crimson")
                continue

            composed = compose_gt_pred(frame_rgb, gt2d, entry["pred"], a_pr)
            ax.imshow(composed)
            if i == 0:
                ax.set_title(f"C={C}", fontweight="bold")

        # Row label (store handle so we can include it in centering)
        t = axes[i, 0].text(
            -0.25, 0.5, f"S={S}%", va="center", ha="right",
            transform=axes[i, 0].transAxes, fontsize=10, fontweight="bold"
        )
        row_label_texts.append(t)

    # -------- Robust suptitle centering over *all* content ----------
    # We include subplot boxes AND the row-label texts to compute the true bounds.
    fig.canvas.draw()  # ensure positions are known
    c3_box = axes[0, 1].get_position(fig)
    c32_box = axes[0, 2].get_position(fig)

# Compute the midpoint between the two columns
    x_mid_c3_c32 = 0.5 * (c3_box.x0 + c32_box.x1)
# Place the title horizontally over prediction block center
    fig.suptitle(regime_title, fontsize=14, fontweight="bold", y=0.95, x=x_mid_c3_c32)
    # ----------------------------------------------------------------

    # Legend bottom-right (inside figure)
    legend_handles = [
        mpatches.Patch(color=(*COLOR_GT, 1.0), label='Ground truth'),
        mpatches.Patch(color=(*COLOR_PR, ALPHA_PR), label='Prediction'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower right',
        bbox_to_anchor=(0.99, 0.02),
        frameon=True, facecolor="white", fontsize=9
    )

    return fig

# =========================
# Main
# =========================
if __name__ == "__main__":
    MODEL_CONFIGS = [
        # IID (S=20,30,60 × C=3,32,64)
        {"label": "IID C3 S20",  "path": "Results/runs/Run_UNET_IID20%_fedWSOcompC3_20250604/Server/results/fedWSOcomp_BestRound15.pth"},
        {"label": "IID C32 S20", "path": "Results/runs/Run_UNET_IID20%_fedWSOcompC32_20250603/Server/results/fedWSOcomp_BestRound16.pth"},
        {"label": "IID C64 S20", "path": "Results/runs/Run_UNET_IID20%_fedWSOcompC64_20250603/Server/results/fedWSOcomp_BestRound21.pth"},
        {"label": "IID C3 S30",  "path": "Results/runs/Run_UNET_iid_30%fedWSOcomp_20250526/Server/results/fedWSOcomp_BestRound24.pth"},
        {"label": "IID C32 S30", "path": "Results/runs/Run_UNET_IID30%_fedWSOcompC32_20250602/Server/results/fedWSOcomp_BestRound21.pth"},
        {"label": "IID C64 S30", "path": "Results/runs/Run_UNET_IID30%_fedWSOcompC64_20250604/Server/results/fedWSOcomp_BestRound19.pth"},
        {"label": "IID C3 S60",  "path": "Results/runs/Run_UNET_IID60%_fedWSOcompC3_20250609/Server/results/fedWSOcomp_BestRound13.pth"},
        {"label": "IID C32 S60", "path": "Results/runs/Run_UNET_IID60%_fedWSOcompC32_20250610/Server/results/fedWSOcomp_BestRound25.pth"},
        {"label": "IID C64 S60", "path": "Results/runs/Run_UNET_IID60%_fedWSOcompC64_20250610/Server/results/fedWSOcomp_BestRound24.pth"},

        # Non-IID
        {"label": "NonIID C3 S20",  "path": "Results/runs/Run_UNET_Non-iid20%_fedWSOcomp_20250528/Server/results/fedWSOcomp_BestRound4.pth"},
        {"label": "NonIID C32 S20", "path": "Results/runs/Run_UNET_Non-iid20%_fedWSOcompC32_20250529/Server/results/fedWSOcomp_BestRound5.pth"},
        {"label": "NonIID C64 S20", "path": "Results/runs/Run_UNET_Non-iid20%_fedWSOcompC64_20250530/Server/results/fedWSOcomp_BestRound5.pth"},
        {"label": "NonIID C3 S30",  "path": "Results/runs/Run_UNET_Non-iid30%_fedWSOcomp_C320250527/Server/results/fedWSOcomp_BestRound7.pth"},
        {"label": "NonIID C32 S30", "path": "Results/runs/Run_UNET_Non-iid30%_fedWSOcompC32_20250601/Server/results/fedWSOcomp_BestRound5.pth"},
        {"label": "NonIID C64 S30", "path": "Results/runs/Run_UNET_Non-iid30%_fedWSOcompC64_20250601/Server/results/fedWSOcomp_BestRound5.pth"},
        {"label": "NonIID C3 S60",  "path": "Results/runs/Run_UNET_nonIID60%_fedWSOcompC3_20250609/Server/results/fedWSOcomp_BestRound7.pth"},
        {"label": "NonIID C32 S60", "path": "Results/runs/Run_UNET_nonIID60%_fedWSOcompC32_20250608/Server/results/fedWSOcomp_BestRound4.pth"},
        {"label": "NonIID C64 S60", "path": "Results/runs/Run_UNET_nonIID60%_fedWSOcompC64_20250608/Server/results/fedWSOcomp_BestRound25.pth"},
    ]

    DATA_PATH = "/home/jovyan/FeTS2022/MICCAI_FeTS2022_TrainingData"
    SERVER_CSV_PATH = "/home/jovyan/FeTSxFedWSO/ServerTestset.csv"

    server_ds = BrainTumorSegmentationCustomDataset(
        csv_file=SERVER_CSV_PATH,
        root_dir=DATA_PATH,
        transforms=val_transform,
        device=DEVICE,
        is_server=True
    )

    # Reference for slice selection (consistent across figures)
    REF_PATH = MODEL_CONFIGS[0]["path"]

    # Choose samples (e.g., 3 best)
    scores = score_all_samples_by_dice(server_ds, REF_PATH)
    worst, median, best = select_k_worst_median_best(scores, k=K_PER_CATEGORY)
    chosen = best[:3]

    # Split configs
    iid_configs     = [m for m in MODEL_CONFIGS if m["label"].startswith("IID")]
    noniid_configs  = [m for m in MODEL_CONFIGS if m["label"].startswith("NonIID")]

    for idx in chosen:
        fig_iid = make_matrix_figure(iid_configs,  server_ds, idx, REF_PATH,
                                     for_pdf=True, regime_title="IID")
        fig_iid.savefig(os.path.join(RESULT_DIR, f"IID_matrix_sample{idx}.pdf"), dpi=FIG_DPI)
        plt.close(fig_iid)

        fig_non = make_matrix_figure(noniid_configs, server_ds, idx, REF_PATH,
                                     for_pdf=True, regime_title="Non-IID")
        fig_non.savefig(os.path.join(RESULT_DIR, f"NonIID_matrix_sample{idx}.pdf"), dpi=FIG_DPI)
        plt.close(fig_non)

    print("[Done] Saved IID and Non-IID matrix PDFs to:", RESULT_DIR)
