"""
Test script to verify FedWSOComp encoding/decoding and analytical
compression ratio on the 3D UNet used in FeTSxFedWSO.

Usage (inside torchenv):

  # 1) On randomly initialized UNet
  python test_fedwsocomp_compression.py

  # 2) On a trained checkpoint
  python test_fedwsocomp_compression.py --ckpt runs/.../model_final.pth

You can adjust sparsity and n_clusters via CLI or .env:

  --sparsity_conv   (default: SPARSITY_CONV or 0.60)
  --sparsity_linear (default: SPARSITY_LINEAR or 0.60)
  --sparsity_bias   (default: SPARSITY_BIAS or 0.00)
  --n_clusters      (default: N_CLUSTERS or 3)
"""

import os
import argparse
import numpy as np
import torch
from dotenv import load_dotenv
from pathlib import Path
from monai.networks.nets import UNet

from utils import (
    encode_weights,
    decode_weights,
    fedwsocomp_top_z,
)

# Optional: fallback if compression_summary is not present
try:
    from utils import compute_compression_ratio  # may or may not exist in your utils
except ImportError:
    compute_compression_ratio = None


def build_unet(device: str = "cpu") -> torch.nn.Module:
    """Build the same 3D UNet as in your FL code."""
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.2,
    )
    return model.to(device)


def infer_layer_type_from_shape(shape) -> str:
    """Match the same logic used inside encode_weights."""
    if len(shape) in (4, 5):
        return "Conv"
    elif len(shape) == 2:
        return "Linear"
    else:
        return "Bias"  # includes 1D and BN params


def main():
    parser = argparse.ArgumentParser(
        description="Verify FedWSOComp encode/decode symmetry and compression."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to UNet checkpoint (.pth). If not provided, uses random init.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to build the model on (cuda/cpu).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of KMeans clusters. If None, read N_CLUSTERS from .env or use 3.",
    )
    parser.add_argument(
        "--sparsity_conv",
        type=float,
        default=None,
        help="Top-z sparsity for Conv layers. If None, read SPARSITY_CONV or 0.20.",
    )
    parser.add_argument(
        "--sparsity_linear",
        type=float,
        default=None,
        help="Top-z sparsity for Linear layers. If None, read SPARSITY_LINEAR or 0.20.",
    )
    parser.add_argument(
        "--sparsity_bias",
        type=float,
        default=None,
        help="Top-z sparsity for Bias layers. If None, read SPARSITY_BIAS or 0.00.",
    )

    args = parser.parse_args()

    # ---------- Load .env for defaults ----------
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    # Resolve hyperparameters (CLI > .env > default)
    n_clusters = (
        args.n_clusters
        if args.n_clusters is not None
        else int(os.getenv("N_CLUSTERS", "3"))
    )

    sparsity_conv = (
        args.sparsity_conv
        if args.sparsity_conv is not None
        else float(os.getenv("SPARSITY_CONV", "0.20"))
    )
    sparsity_linear = (
        args.sparsity_linear
        if args.sparsity_linear is not None
        else float(os.getenv("SPARSITY_LINEAR", "0.20"))
    )
    sparsity_bias = (
        args.sparsity_bias
        if args.sparsity_bias is not None
        else float(os.getenv("SPARSITY_BIAS", "0.0"))
    )

    sparsity_cfg = {
        "Conv": sparsity_conv,
        "Linear": sparsity_linear,
        "Bias": sparsity_bias,
        "BatchNorm": 0.0,
    }

    print("====================================================")
    print(" FedWSOComp Verification Script")
    print("----------------------------------------------------")
    print(f" Device          : {args.device}")
    print(f" Checkpoint      : {args.ckpt}")
    print(f" N_CLUSTERS      : {n_clusters}")
    print(f" Sparsity Conv   : {sparsity_conv}")
    print(f" Sparsity Linear : {sparsity_linear}")
    print(f" Sparsity Bias   : {sparsity_bias}")
    print("====================================================\n")

    # ---------- Build UNet ----------
    device = torch.device(args.device)
    model = build_unet(device=device)

    # ---------- Load checkpoint if provided ----------
    if args.ckpt is not None:
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        print(f"[INFO] Loading checkpoint from {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
    else:
        print("[INFO] No checkpoint provided. Using random initialization.")

    model.eval()

    # ---------- Extract weights as list[np.ndarray] ----------
    state_dict = model.state_dict()
    layer_names = sorted(state_dict.keys())
    weights = [state_dict[k].detach().cpu().numpy().astype(np.float32) for k in layer_names]

    # ---------- Encode with FedWSOComp ----------
    print("[INFO] Running encode_weights...")
    encoded = encode_weights(weights, sparsity_cfg=sparsity_cfg, n_clusters=n_clusters)

    # ---------- Analytical compression stats ----------
    summary = encoded.get("compression_summary", None)

    if summary is not None:
        orig_MB = float(summary.get("orig_mb", 0.0))
        comp_MB = float(summary.get("compressed_mb", 0.0))
        cr = float(summary.get("ratio", 1.0))
    else:
        # Fallback: use compute_compression_ratio if available
        if compute_compression_ratio is not None:
            stats = compute_compression_ratio(encoded, bits_per_param=32)
            orig_MB = stats["orig_bits"] / 8.0 / (1024**2)
            comp_MB = stats["compressed_bits"] / 8.0 / (1024**2)
            cr = stats["compression_ratio"]
        else:
            # Very rough fallback: dense FP32 as orig, no estimate for comp
            total_params = sum(int(np.prod(w.shape)) for w in weights)
            orig_MB = (total_params * 32) / 8.0 / (1024**2)
            comp_MB = orig_MB  # unknown
            cr = 1.0

    print("\n================ Compression Summary ================")
    print(f" Original dense size (FP32) : {orig_MB:.3f} MB")
    print(f" Estimated compressed size  : {comp_MB:.3f} MB")
    print(f" Compression ratio (orig/comp) : {cr:.2f}x")
    print("====================================================\n")

    # ---------- Decode and compare to sparsified baseline ----------
    print("[INFO] Running decode_weights...")
    decoded = decode_weights(encoded)

    # For symmetry checking, compare decoded weights against the
    # sparsified baseline (Top-z applied to original), since
    # encode_weights internally does: w -> Top-z -> KMeans.
    per_layer_max_diff = []
    per_layer_l2_diff = []

    print("============= Per-layer Reconstruction Error =========")
    for idx, (name, w_orig, w_dec) in enumerate(zip(layer_names, weights, decoded)):
        layer_type = infer_layer_type_from_shape(w_orig.shape)
        z = sparsity_cfg.get(layer_type, 0.0)

        # Sparsified baseline for this layer
        w_sparse = fedwsocomp_top_z(w_orig, sparsity=z)

        # Differences between decoded and sparsified baseline
        diff = w_dec.astype(np.float32) - w_sparse.astype(np.float32)
        max_abs = float(np.max(np.abs(diff)))
        l2 = float(np.linalg.norm(diff.ravel()))

        per_layer_max_diff.append(max_abs)
        per_layer_l2_diff.append(l2)

        print(
            f"[Layer {idx:03d}] {name:40s} "
            f"type={layer_type:7s}  max|Δ|={max_abs:.6e}  L2={l2:.6e}"
        )

    # ---------- Global error summary ----------
    all_max = float(np.max(per_layer_max_diff))
    total_l2 = float(np.sqrt(np.sum(np.array(per_layer_l2_diff) ** 2)))

    print("\n================ Global Reconstruction Error =========")
    print(f" Max per-layer max|Δ|   : {all_max:.6e}")
    print(f" Global L2 (across all layers) : {total_l2:.6e}")
    print("=====================================================\n")

    print("Done. You can now report:")
    print("  - analytical compression ratio (MB, x-times),")
    print("  - reconstruction error (max|Δ|, L2) to show near-lossless behaviour.")
    print("These directly support the 'communication efficiency' and 'fidelity' claims.")


if __name__ == "__main__":
    main()
