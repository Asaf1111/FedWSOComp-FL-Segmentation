import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch

def to_numpy(data):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def visualize_brats_segmentation(images, labels, preds=None, batch_index=0, slice_index=None,
                                 background_channel="T1ce", save_fig=False, save_dir=".", prefix="case"):
    images_np = to_numpy(images)
    labels_np = to_numpy(labels)
    preds_np = to_numpy(preds)

    if images_np.ndim == 5:
        images_np = images_np[batch_index]
    if labels_np is not None and labels_np.ndim == 5:
        labels_np = labels_np[batch_index]
    if preds_np is not None and preds_np.ndim == 5:
        preds_np = preds_np[batch_index]

    # Get modality index (default T1ce)
    modality_names = ["T1", "T2", "FLAIR", "T1ce"]
    bg_index = modality_names.index(background_channel) if background_channel in modality_names else 3

    # Prepare WT masks from label and prediction
    label_map = labels_np[0] if labels_np.ndim == 4 and labels_np.shape[0] == 1 else labels_np
    WT_mask = label_map.astype(bool)

    if slice_index is None:
        if WT_mask.any():
            slice_index = int(np.argmax(WT_mask.sum(axis=(0, 1))))
        else:
            slice_index = images_np.shape[-1] // 2

    # Normalize background modality
    bg_slice = images_np[bg_index, :, :, slice_index]
    bg_display = (bg_slice - bg_slice.min()) / (bg_slice.max() - bg_slice.min() + 1e-8)
    background_rgb = np.stack([bg_display] * 3, axis=-1)

    def build_wt_overlay(mask):
        if mask.ndim == 3:
            mask = mask[:, :, slice_index]
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[mask, 1] = 1.0  # Green for WT
        overlay[mask, 3] = 0.4  # Alpha transparency
        return overlay

    overlay_gt = build_wt_overlay(WT_mask)
    overlay_pred = None

    if preds_np is not None:
        pred_map = preds_np[0] if preds_np.ndim == 4 and preds_np.shape[0] == 1 else preds_np
        pred_mask = pred_map > 0.5
        overlay_pred = build_wt_overlay(pred_mask)

    # ---- Plot input modalities ---- #
    fig1, axes1 = plt.subplots(1, 4, figsize=(12, 3))
    for i, modality in enumerate(modality_names):
        axes1[i].imshow(images_np[i, :, :, slice_index], cmap='gray')
        axes1[i].set_title(modality)
        axes1[i].axis('off')
    fig1.suptitle(f"Input Modalities (Slice {slice_index})", fontsize=14)
    fig1.tight_layout()

    # ---- Plot GT and Prediction Overlays ---- #
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    axes2[0].imshow(background_rgb)
    axes2[0].imshow(overlay_gt)
    axes2[0].set_title("Ground Truth WT")
    axes2[0].axis('off')

    if overlay_pred is not None:
        axes2[1].imshow(background_rgb)
        axes2[1].imshow(overlay_pred)
        axes2[1].set_title("Predicted WT")
        axes2[1].axis('off')
    else:
        axes2[1].axis('off')

    legend_patches = [Patch(color='green', label='Whole Tumor (WT)')]
    axes2[1].legend(handles=legend_patches, loc='upper right')
    fig2.suptitle(f"Overlay Comparison (Slice {slice_index})", fontsize=14)
    fig2.tight_layout()

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, f"{prefix}_slice{slice_index}_inputs.png"))
        fig2.savefig(os.path.join(save_dir, f"{prefix}_slice{slice_index}_overlay.png"))

    return fig1, fig2
