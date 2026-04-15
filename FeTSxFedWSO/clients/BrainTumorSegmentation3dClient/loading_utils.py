import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.meta_tensor import MetaTensor
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped
)

class BrainTumorSegmentationCustomDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transforms=None, device="cuda:0",
        val_perc=0.1, test_perc=0.1, mode="train", is_server=False,
        random_state=42
    ):
        self.client_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.is_server = is_server
        self.random_state = random_state
        self.device = device

        self.loading_transforms = Compose([
            LoadImaged(keys=["t1", "flair", "t2", "t1ce", "seg"]),
            NormalizeIntensityd(keys=["t1", "flair", "t2", "t1ce"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["t1", "flair", "t2", "t1ce", "seg"], track_meta=False),

        ])

        self._prepare_splits(val_perc, test_perc)
        if self.is_server:
            self.indices = list(range(len(self.client_csv)))
            self.mode = "test"
            print(f"[DEBUG] Server dataset length: {len(self.indices)}") 
            return

    def _prepare_splits(self, val_perc, test_perc):
        if self.is_server:
            self.indices = list(range(len(self.client_csv)))
            self.mode = "test"
            return

        train_val_idx, test_idx = train_test_split(
            range(len(self.client_csv)),
            test_size=test_perc,
            random_state=self.random_state
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_perc / (1 - test_perc),
            random_state=self.random_state
        )

        if self.mode == "train":
            self.indices = train_idx
        elif self.mode == "val":
            self.indices = val_idx
        elif self.mode == "test":
            self.indices = test_idx
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train', 'val' or 'test")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        actual_idx = self.indices[idx]
        sample_id = str(self.client_csv.iloc[actual_idx, 0])
        sample_path = os.path.join(self.root_dir, sample_id)

        sample_dict = {
            "t1": os.path.join(sample_path, f"{sample_id}_t1.nii.gz"),
            "flair": os.path.join(sample_path, f"{sample_id}_flair.nii.gz"),
            "t1ce": os.path.join(sample_path, f"{sample_id}_t1ce.nii.gz"),
            "t2": os.path.join(sample_path, f"{sample_id}_t2.nii.gz"),
            "seg": os.path.join(sample_path, f"{sample_id}_seg.nii.gz"),
        }

        sample_volumes = self.loading_transforms(sample_dict)

        input_image = torch.cat([
            sample_volumes["t1"].unsqueeze(0),
            sample_volumes["t1ce"].unsqueeze(0),
            sample_volumes["t2"].unsqueeze(0),
            sample_volumes["flair"].unsqueeze(0),
        ], dim=0)
        label = sample_volumes["seg"]
        if label.ndim == 3:
            label = label.unsqueeze(0) 

        input_image = MetaTensor(input_image.clone(), meta={"affine": torch.eye(4), "original_channel_dim": 0})
        label = MetaTensor(label.clone(), meta={"affine": torch.eye(4), "original_channel_dim": 0})
        final_dict = {"image": input_image, "label": label}
        print(f"[DEBUG][Dataset] Before transform: image={final_dict['image'].shape}, label={final_dict['label'].shape}")    

        if idx == 0:
            print(f"[DEBUG] Sample {sample_id} — BEFORE transform")
            print(f"[DEBUG] Input shape: {input_image.shape}")
            print(f"[DEBUG] Label shape: {label.shape}")
            print(f"[DEBUG] Label unique values: {torch.unique(label)}")
            print(f"[DEBUG][Dataset] Pre-transform shapes: image={final_dict['image'].shape}, label={final_dict['label'].shape}")

        if self.transforms is not None:
            transformed_sample = self.transforms(final_dict)
        else:
            transformed_sample = final_dict

        if idx == 0:
            print(f"[DEBUG] After transforms — image: {transformed_sample['image'].shape}, label: {transformed_sample['label'].shape}")
        return transformed_sample    


class BrainTumorSegmentationCustomDatasetExtended(Dataset):
    def __init__(
        self, csv_file, root_dir, transforms=None, device="cuda:0",
        val_perc=0.1, test_perc=0.1, mode="train", is_server=False,
        random_state=42
    ):
        self.client_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.is_server = is_server
        self.random_state = random_state
        self.device = device

        self.loading_transforms = Compose([
            LoadImaged(keys=["t1", "flair", "t2", "t1ce", "seg"]),
            NormalizeIntensityd(keys=["t1", "flair", "t2", "t1ce"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["t1", "flair", "t2", "t1ce", "seg"], track_meta=False),

        ])

        self._prepare_splits(val_perc, test_perc)
        if self.is_server:
            self.indices = list(range(len(self.client_csv)))
            self.mode = "test"
            print(f"[DEBUG] Server dataset length: {len(self.indices)}") 
            return

    def _prepare_splits(self, val_perc, test_perc):
        if self.is_server:
            self.indices = list(range(len(self.client_csv)))
            self.mode = "test"
            return

        train_val_idx, test_idx = train_test_split(
            range(len(self.client_csv)),
            test_size=test_perc,
            random_state=self.random_state
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_perc / (1 - test_perc),
            random_state=self.random_state
        )

        if self.mode == "train":
            self.indices = train_idx
        elif self.mode == "val":
            self.indices = val_idx
        elif self.mode == "test":
            self.indices = test_idx
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train', 'val' or 'test")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        actual_idx = self.indices[idx]
        sample_id = str(self.client_csv.iloc[actual_idx, 0])
        sample_path = os.path.join(self.root_dir, sample_id)

        sample_dict = {
            "t1": os.path.join(sample_path, f"{sample_id}_t1.nii.gz"),
            "flair": os.path.join(sample_path, f"{sample_id}_flair.nii.gz"),
            "t1ce": os.path.join(sample_path, f"{sample_id}_t1ce.nii.gz"),
            "t2": os.path.join(sample_path, f"{sample_id}_t2.nii.gz"),
            "seg": os.path.join(sample_path, f"{sample_id}_seg.nii.gz"),
        }

        sample_volumes = self.loading_transforms(sample_dict)

        input_image = torch.cat([
            sample_volumes["t1"].unsqueeze(0),
            sample_volumes["t1ce"].unsqueeze(0),
            sample_volumes["t2"].unsqueeze(0),
            sample_volumes["flair"].unsqueeze(0),
        ], dim=0)
        label = sample_volumes["seg"]
        if label.ndim == 3:
            label = label.unsqueeze(0) 

        input_image = MetaTensor(input_image.clone(), meta={"affine": torch.eye(4), "original_channel_dim": 0})
        label = MetaTensor(label.clone(), meta={"affine": torch.eye(4), "original_channel_dim": 0})
        final_dict = {"image": input_image, "label": label}
        print(f"[DEBUG][Dataset] Before transform: image={final_dict['image'].shape}, label={final_dict['label'].shape}")    

        if idx == 0:
            print(f"[DEBUG] Sample {sample_id} — BEFORE transform")
            print(f"[DEBUG] Input shape: {input_image.shape}")
            print(f"[DEBUG] Label shape: {label.shape}")
            print(f"[DEBUG] Label unique values: {torch.unique(label)}")
            print(f"[DEBUG][Dataset] Pre-transform shapes: image={final_dict['image'].shape}, label={final_dict['label'].shape}")

        if self.transforms is not None:
            transformed_sample = self.transforms(final_dict)
        else:
            transformed_sample = final_dict

        if idx == 0:
            print(f"[DEBUG] After transforms — image: {transformed_sample['image'].shape}, label: {transformed_sample['label'].shape}")
        return transformed_sample, sample_id    
