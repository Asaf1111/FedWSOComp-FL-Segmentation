
import os
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from visualize_brats_debug import visualize_brats_segmentation
import torch

def visualize_server_predictions(model, loader, save_dir="server_visuals", device="cuda:0"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = sliding_window_inference(inputs, roi_size=(240, 240, 160), sw_batch_size=1, predictor=model)

            image_np = inputs.cpu()
            label_np = labels.cpu()
            output_np = outputs.cpu()

            visualize_brats_segmentation(
                images=image_np,
                labels=label_np,
                preds=output_np,
                batch_index=0,
                save_fig=True,
                save_dir=save_dir,
                prefix=f"server_case{idx}"
            )

            if idx == 65:  # Visualize first 5 test cases
                break

