import os

import csv
import torch
import numpy as np
import time
from monai.data import decollate_batch, pad_list_data_collate
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.transforms import DivisiblePadd
from monai.transforms import MapTransform
from monai.utils import set_determinism
from monai.transforms import (
    Activationsd, AsDiscreted, Compose, Invertd, NormalizeIntensityd,
    Orientationd, EnsureTyped, Spacingd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, EnsureChannelFirstd, CropForegroundd
)
from monai.data.meta_tensor import MetaTensor
from monai.metrics import DiceMetric




set_determinism(seed=42)

class ConvertToWholeTumorOnlyd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if label.shape[0] == 1:
                label = label[0]
            # Always convert to binary WT regardless of label state
            wt = torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4).float()
            d[key] = wt.unsqueeze(0)
        return d


train_transform = Compose([
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    DivisiblePadd(keys=["image", "label"], k=32),  
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ConvertToWholeTumorOnlyd(keys=["label"]),
    EnsureTyped(keys=["image", "label"]),
])

val_transform = Compose([
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    DivisiblePadd(keys=["image", "label"], k=32),
    ConvertToWholeTumorOnlyd(keys=["label"]),
    EnsureTyped(keys=["image", "label"]),
])

post_transforms_with_invert = Compose([
    Invertd(
        keys="pred",
        transform=val_transform,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device="cpu",
    ),
    Activationsd(keys="pred", sigmoid=True),
    AsDiscreted(keys="pred", threshold=0.5),
    ConvertToWholeTumorOnlyd(keys=["label"]),
    AsDiscreted(keys="label", threshold=0.5)
])

post_transforms_no_invert = Compose([
    ConvertToWholeTumorOnlyd(keys=["label"]),
    Activationsd(keys="pred", sigmoid=True),
    AsDiscreted(keys="pred", threshold=0.5),
    AsDiscreted(keys="label", threshold=0.5),
])

def inference(input, model, VAL_AMP):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
    if VAL_AMP:
        with torch.autocast("cuda"):
            return _compute(input)
    else:
        return _compute(input)

def train(model, loader, device, optimizer, epoch, loss_function, scaler, logger, lr_scheduler):
    model.train()
    epoch_loss = 0
    step = 0
    step_times = []
    for batch_data in loader:
        step_start = time.time()
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            print(inputs.size())
            outputs = model(inputs)
            if epoch == 0 and step == 1:
                print(f"[DEBUG] Output stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
                print(f"[DEBUG] Label stats: min={labels.min().item()}, max={labels.max().item()}")    
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        step_time = time.time() - step_start
        epoch_loss += loss.item()
        step_times.append(step_time)
        print(f"[Train][Epoch {epoch+1}][Step {step}] Loss: {loss.item():.4f} | Time: {step_time:.2f}s")
    avg_loss = epoch_loss / step
    logger.info(f"[TRAIN] Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    print(f"[Train][Epoch {epoch+1}] Average Loss: {avg_loss:.4f} | Avg Step Time: {sum(step_times)/len(step_times):.2f}s")
    return avg_loss
def validation(model, epoch, loader, device, post_transforms, dice_metric, dice_metric_batch):
    best_metric = -1
    best_metric_epoch = -1
    model.eval()
    step = 0
    with torch.no_grad():
        for data in loader:
            step += 1
            start_time = time.time()
            val_inputs = data["image"].to(device)
            val_labels = data["label"].to(device)
            val_outputs = inference(val_inputs, model, VAL_AMP=False)
            val_outputs = [post_transforms({"pred": i, "label": val_labels[j]})["pred"] for j, i in enumerate(decollate_batch(val_outputs))]
            val_labels = [i.to(device) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)
            step_duration = time.time() - start_time
            print(f"[Validation] Epoch {epoch+1}, Step {step}, Step Time: {step_duration:.2f}s")

        mean_dice = 0.0
        try:
            mean_dice = dice_metric_batch.aggregate().item()
        except Exception as e:
            print(f"[ERROR] Validation Dice aggregation failed: {e}")

        dice_metric.reset()
        dice_metric_batch.reset()

        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1

        print(f"[Validation][Epoch {epoch + 1}] Mean Dice: {mean_dice:.4f}")
        return {"mean_dice": mean_dice}


import os
import csv
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

def test(model, loader, post_transforms, device, dice_metric_batch=None, logger=None, save_csv=False, csv_path=None):
    model.eval()
    detailed_logs = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            print(f"[DEBUG] Inference input shape: {inputs.shape}")

            outputs = sliding_window_inference(
                inputs, roi_size=inputs.shape[2:], sw_batch_size=1, predictor=model
            )

            outputs = decollate_batch(outputs)
            labels = decollate_batch(labels)

            for l in labels:
                print(f"[DEBUG] [SERVER] Label Foreground Voxel Count: {torch.sum(l == 1).item()}")

            print(f"[DEBUG] Label Foreground Voxel Count: {torch.sum(labels[0] == 1).item()}")

            data = [post_transforms({"pred": p, "label": l}) for p, l in zip(outputs, labels)]

            for i, sample in enumerate(data):
                pred = sample["pred"]
                label = sample["label"]

                intersection = torch.sum((pred == 1) & (label == 1)).item()
                union = torch.sum((pred == 1) | (label == 1)).item()
                dice = 1.0 if union == 0 and intersection == 0 else 2 * intersection / (union + 1e-8)
                dice = float(min(max(dice, 0.0), 1.0))

                # Clamp Dice to [0, 1]
                dice = min(max(dice, 0.0), 1.0)

                print(f"[DEBUG] Unique values in label: {torch.unique(label)}")
                print(f"[DEBUG] Unique values in pred: {torch.unique(pred)}")
                print(f"[TEST DEBUG] Batch {batch_idx + 1} | Sample {i} | Dice: {dice:.4f}")

                detailed_logs.append({
                    "batch": batch_idx + 1,
                    "sample": i,
                    "foreground_pred": int(torch.sum(pred == 1)),
                    "foreground_label": int(torch.sum(label == 1)),
                    "intersection": int(intersection),
                    "union": int(union),
                    "dice": round(dice, 4),
                })

            # Aggregate all predictions and labels in the batch
            preds = torch.stack([d["pred"] for d in data])
            labels = torch.stack([d["label"] for d in data])

            print(f"[DEBUG] Aggregated preds unique values: {torch.unique(preds)}")
            print(f"[DEBUG] Aggregated labels unique values: {torch.unique(labels)}")

            if preds.shape[1] == 1:
                preds = preds[:, 0]
                labels = labels[:, 0]

            print(f"[DEBUG] Metric input shapes -> preds: {preds.shape}, labels: {labels.shape}")

            if logger:
                logger.debug(f"[TEST] Batch {batch_idx + 1} | Pred shape: {preds[0].shape} | Label shape: {labels[0].shape}")

    # Final Dice average
    dice_values = [entry["dice"] for entry in detailed_logs if "dice" in entry]
    mean_dice = round(np.mean(dice_values), 4) if dice_values else 0.0

    print(f"[TEST DEBUG] Final aggregated mean Dice: {mean_dice:.4f}")

    # Save CSV if required
    if save_csv and csv_path and len(detailed_logs) > 0:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detailed_logs[0].keys())
            writer.writeheader()
            writer.writerows(detailed_logs)
        print(f"[INFO] Saved per-sample Dice to {csv_path}")

    return {"mean_dice": mean_dice, "logs": detailed_logs}


__all__ = [
    "train", "test", "validation",
    "post_transforms_with_invert", "post_transforms_no_invert",
    "val_transform", "train_transform",
    "ConvertToWholeTumorOnlyd"
]