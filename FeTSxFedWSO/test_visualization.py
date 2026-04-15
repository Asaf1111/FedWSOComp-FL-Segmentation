import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.data import DataLoader
from clients.BrainTumorSegmentation3dClient.loading_utils import BrainTumorSegmentationCustomDataset
from clients.BrainTumorSegmentation3dClient.utils import val_transform
from visualize_brats_debug import visualize_brats_segmentation

# ------------------------ Paths ------------------------ #
CSV_PATH = "/home/jovyan/FeTSxFedWSO/data_splitting/clients/iid/client2_iid_dataset.csv"
DATA_PATH = "/home/jovyan/FeTS2022/MICCAI_FeTS2022_TrainingData"

# ------------------------ Dataset ------------------------ #
dataset = BrainTumorSegmentationCustomDataset(
    csv_file=CSV_PATH,
    root_dir=DATA_PATH,
    transforms=val_transform,
    mode="test",
    is_server=False,
)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# ------------------------ UNet Model ------------------------ #
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='INSTANCE',
    dropout=0.2
).cuda()
model.eval()

# ------------------------ Visualization Directory ------------------------ #
save_dir = "vis_check_client2_full"
os.makedirs(save_dir, exist_ok=True)

# ------------------------ Visualize Entire Client 2 ------------------------ #
for i, batch in enumerate(loader):
    image = batch["image"].cuda()
    label = batch["label"].cpu()

    with torch.no_grad():
        output = model(image)
        output = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1, keepdim=True).cpu()

    # --- SANITY CHECK for prediction ---
    unique_vals = torch.unique(pred)
    print(f"[INFO] Sample {i} | Unique predicted labels: {unique_vals.tolist()}")
    if unique_vals.max() == 0:
        print(f"[WARNING]  Sample {i} has empty prediction. Model might be underfitting.")

    # Determine informative slice
    wt_mask = (label > 0).float()
    try:
        if wt_mask.shape[1] == 1:
            tumor_sums = wt_mask[0, 0].sum(dim=(0, 1))
        else:
            tumor_sums = wt_mask[0].sum(dim=(1, 2)).sum(dim=0)
        slice_index = int(torch.argmax(tumor_sums).item()) if tumor_sums.sum() > 0 else label.shape[-1] // 2

        visualize_brats_segmentation(
            images=batch["image"],
            labels=label,
            preds=pred,
            batch_index=0,
            slice_index=slice_index,
            background_channel="FLAIR",
            save_fig=True,
            save_dir=save_dir,
            prefix=f"client2_sample{i:03d}"
        )
        print(f" Saved client2_sample{i:03d} at slice {slice_index}")

    except Exception as e:
        print(f"[ERROR] Sample {i} failed: {e}")

print("\n All Client 2 visualizations saved to 'vis_check_client2_full/'")
