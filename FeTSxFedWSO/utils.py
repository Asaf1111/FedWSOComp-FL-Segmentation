# utils.py (root-level: FedWSOComp compression utilities)
# UPDATED: supports FEDAVG / SKH / SH / KH + theoretical comm estimator
# NOTE: This file only changes compression behavior + metrics. Training is untouched.

import os
import numpy as np
import heapq
from collections import Counter, OrderedDict
from sklearn.cluster import KMeans

# (Optional) these are used only by your symmetry tests below
import matplotlib.pyplot as plt
import torch
from monai.networks.nets import UNet


# ============================================================
#  Huffman Dictionary (Paper Eq. 7)
# ============================================================

def build_huffman_dict(symbols):
    counter = Counter(symbols)
    if len(counter) == 1:
        sym = next(iter(counter))
        return OrderedDict([(sym, "0")])

    heap = [[weight, [sym, ""]] for sym, weight in sorted(counter.items())]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return OrderedDict(sorted(heap[0][1:], key=lambda x: x[0]))


# ============================================================
#  Soft Top-z Sparsification (Paper Eq. 5)
#  (keeps dense tensors; clamps small weights to ±w_z)
# ============================================================

def fedwsocomp_top_z(w, sparsity):
    flat_w = w.flatten()
    N = flat_w.size
    z = int(sparsity * N)

    if z == 0 or sparsity <= 0.0:
        return flat_w.copy().reshape(w.shape)

    # z-th largest by magnitude
    sorted_abs = np.sort(np.abs(flat_w))[::-1]
    w_z = sorted_abs[z - 1] if z <= N else sorted_abs[-1]

    # Soft top-z: clamp small weights to ±w_z, do NOT zero them
    new_w = np.where(np.abs(flat_w) > w_z, flat_w, np.sign(flat_w) * w_z)
    return new_w.reshape(w.shape)


# ============================================================
#  Encoding: supports modes FEDAVG / SKH / SH / KH
#  - SKH: Sparsification + KMeans + Huffman
#  - SH : Sparsification + Huffman (NO KMeans)
#  - KH : KMeans + Huffman (NO Sparsification)
#  - FEDAVG: No compression (encode called rarely; safe fallback)
#
#  IMPORTANT:
#   - We keep your current "dense address_table" behavior (all indices),
#     because Option A focuses on theoretical metrics only.
# ============================================================

def encode_weights(weights, sparsity_cfg, n_clusters=3):
    """
    Returns encoded_dict compatible with your current decode_weights().
    """
    ENCODING_MODE = os.getenv("ENCODING_MODE", "SKH").upper()

    centroids, encoded_labels, label_dicts = [], [], []
    address_table, shapes, sparse_indices = [], [], []

    for i, w in enumerate(weights):
        shape = w.shape
        shapes.append(tuple(int(s) for s in shape))

        # Layer type (for sparsity_cfg)
        if len(w.shape) == 5:
            layer_type = "Conv"
        elif len(w.shape) == 2:
            layer_type = "Linear"
        else:
            layer_type = "Bias"

        # Decide sparsity based on mode
        if ENCODING_MODE in ("SKH", "SH"):
            sparsity = float(sparsity_cfg.get(layer_type, 0.0))
        else:
            # KH or FEDAVG -> no sparsification
            sparsity = 0.0

        # --- Soft Top-z Sparsification ---
        w_sparse = fedwsocomp_top_z(w, sparsity)
        flat_vals = w_sparse.flatten().astype(np.float32)

        # Keep your current behavior: all positions are "active"
        sparse_idx = np.arange(flat_vals.size, dtype=np.int64)
        sparse_indices.append(sparse_idx)
        address_table.append(sparse_idx)

        # =====================================================
        # SH MODE: Sparsification + Huffman (NO KMeans)
        # We encode by dictionary-indexing unique values.
        # This is deterministic and decode-compatible.
        # =====================================================
        if ENCODING_MODE == "SH":
            uniq_vals, labels = np.unique(flat_vals, return_inverse=True)
            centroids.append(uniq_vals.astype(np.float32))

            # labels can exceed 255 -> use uint16 safely
            encoded_labels.append(labels.astype(np.uint16))

            label_dicts.append(build_huffman_dict(labels.tolist()))
            continue

        # =====================================================
        # KH / SKH MODE: KMeans + Huffman
        # =====================================================
        if ENCODING_MODE in ("KH", "SKH"):
            k = min(int(n_clusters), len(np.unique(flat_vals)))
            if k < 1:
                centroids.append(np.array([0.0], dtype=np.float32))
                encoded_labels.append(np.zeros_like(flat_vals, dtype=np.uint8))
                label_dicts.append(OrderedDict([(0, "0")]))
                continue

            kmeans = KMeans(n_clusters=k, n_init=1, random_state=0)
            cluster_ids = kmeans.fit_predict(flat_vals.reshape(-1, 1))
            raw_centroids = kmeans.cluster_centers_.flatten()

            # Sort centroids and remap labels for determinism
            sort_idx = np.argsort(raw_centroids)
            sorted_centroids = raw_centroids[sort_idx]
            remap = np.zeros_like(sort_idx)
            remap[sort_idx] = np.arange(len(sort_idx))
            sorted_labels = remap[cluster_ids]

            label_dicts.append(build_huffman_dict(sorted_labels.tolist()))
            centroids.append(sorted_centroids.astype(np.float32))
            encoded_labels.append(sorted_labels.astype(np.uint8))
            continue

        # =====================================================
        # FEDAVG or unknown mode: safe fallback
        # =====================================================
        uniq_vals, labels = np.unique(flat_vals, return_inverse=True)
        centroids.append(uniq_vals.astype(np.float32))
        encoded_labels.append(labels.astype(np.uint16))
        label_dicts.append(build_huffman_dict(labels.tolist()))

    return {
        "centroids": [np.copy(c) for c in centroids],
        "encoded_labels": [np.copy(x) for x in encoded_labels],
        "label_dicts": label_dicts,
        "address_table": address_table,
        "shapes": shapes,
        "mode": "huffman",
        "sparse_indices": sparse_indices,
    }


# ============================================================
#  Decoding (compatible with SH uint16 labels + KMeans uint8 labels)
# ============================================================

def decode_weights(encoded_dict):
    centroids = encoded_dict["centroids"]
    encoded_labels = encoded_dict["encoded_labels"]
    shapes = encoded_dict["shapes"]
    sparse_indices = encoded_dict["sparse_indices"]

    decoded_weights = []

    for i, shape in enumerate(shapes):
        shape = tuple(int(round(s)) for s in shape)
        N = int(np.prod(shape))
        dense = np.zeros(N, dtype=np.float32)

        idxs = np.asarray(sparse_indices[i], dtype=np.int64)

        # labels can be uint8 (KMeans) or uint16 (SH unique-value labels)
        cluster_ids = np.asarray(encoded_labels[i], dtype=np.int64)

        # Safety align
        if cluster_ids.size != idxs.size:
            min_len = min(cluster_ids.size, idxs.size)
            cluster_ids = cluster_ids[:min_len]
            idxs = idxs[:min_len]

        # Map ids to centroids
        c = np.asarray(centroids[i], dtype=np.float32)
        dense[idxs] = c[cluster_ids]

        decoded_weights.append(dense.reshape(shape))

    return decoded_weights


# ============================================================
#  NEW (Option A): Theoretical Communication Bit Estimator
#  - Does NOT change training or wire format.
#  - Used ONLY for logging communication metrics.
# ============================================================

def _layer_type_from_shape(shape):
    if len(shape) in (5, 4):
        return "Conv"
    if len(shape) == 2:
        return "Linear"
    return "Bias"


def estimate_baseline_bits_from_shapes(shapes, bits_per_param=32):
    """Baseline (FedAvg) bits if sending all params as FP32."""
    total = 0
    for shape in shapes:
        shape = tuple(int(round(s)) for s in shape)
        total += int(np.prod(shape)) * int(bits_per_param)
    return int(total)


def estimate_comm_bits_theoretical(encoded_dict, sparsity_cfg, bits_per_param=32):
    """
    Paper-style theoretical comm estimate.
    We compute layerwise:
      N = total params
      s = sparsity from sparsity_cfg by layer type
      N_eff = (1 - s) * N

      bits_centroids = k * 32
      bits_labels    = N_eff * avg_huffman_code_len
      bits_indices   = N_eff * ceil(log2(N))

    NOTE: This is theoretical; it intentionally ignores Python pickle overhead.
    """
    shapes = encoded_dict["shapes"]
    centroids_list = encoded_dict["centroids"]
    encoded_labels_list = encoded_dict["encoded_labels"]
    label_dicts_list = encoded_dict["label_dicts"]

    total_bits = 0.0

    for shape, centroids, labels, hdict in zip(
        shapes, centroids_list, encoded_labels_list, label_dicts_list
    ):
        shape = tuple(int(round(s)) for s in shape)
        N = int(np.prod(shape))
        if N <= 0:
            continue

        layer_type = _layer_type_from_shape(shape)
        s = float(sparsity_cfg.get(layer_type, 0.0))
        s = max(0.0, min(0.9999, s))
        N_eff = int(round((1.0 - s) * N))
        N_eff = max(0, min(N, N_eff))

        # Centroids
        k = len(centroids) if centroids is not None else 0
        bits_centroids = k * int(bits_per_param)

        if N_eff == 0:
            total_bits += bits_centroids
            continue

        # Avg Huffman code length from dict + actual label frequencies
        labels = np.asarray(labels, dtype=np.int64)
        if labels.size > 0 and isinstance(hdict, (dict, OrderedDict)) and len(hdict) > 0:
            counts = Counter(labels.tolist())
            code_len = {sym: len(code) for sym, code in hdict.items()}
            fallback_len = max(code_len.values()) if code_len else 1

            total_code_len = 0
            total_count = 0
            for sym, cnt in counts.items():
                L = code_len.get(sym, fallback_len)
                total_code_len += L * cnt
                total_count += cnt

            avg_code_len = (total_code_len / total_count) if total_count > 0 else float(fallback_len)
        else:
            avg_code_len = 1.0

        bits_labels = float(N_eff) * float(avg_code_len)

        # Ideal index bits
        bits_per_index = int(np.ceil(np.log2(N))) if N > 1 else 1
        bits_indices = int(N_eff) * int(bits_per_index)

        total_bits += float(bits_centroids) + float(bits_labels) + float(bits_indices)

    return float(total_bits)


# ============================================================
#  Symmetry Tests & Visualization (kept; will still work)
# ============================================================

def fedwsocomp_symmetry_test():
    print("\n===== FedWSOcomp Symmetry Test =====")
    np.random.seed(1234)

    weights = [
        np.random.randn(32).astype(np.float32),            # Bias
        np.random.randn(8, 3, 3, 3, 3).astype(np.float32)  # Conv
    ]
    sparsity_cfg = {"Conv": 0.5, "Linear": 0.5, "Bias": 0.2}
    n_clusters = 3

    # Stage 1: Sparsify + Quantize (KMeans)
    quantized_list = []
    for w in weights:
        if len(w.shape) == 5:
            typ = "Conv"
        elif len(w.shape) == 2:
            typ = "Linear"
        else:
            typ = "Bias"

        w_sparse = fedwsocomp_top_z(w, sparsity_cfg[typ])
        flat_vals = w_sparse.flatten()

        k = min(n_clusters, len(np.unique(flat_vals)))
        kmeans = KMeans(n_clusters=k, n_init=1, random_state=0)
        cluster_ids = kmeans.fit_predict(flat_vals.reshape(-1, 1))
        raw_centroids = kmeans.cluster_centers_.flatten()

        sort_idx = np.argsort(raw_centroids)
        sorted_centroids = raw_centroids[sort_idx]
        remap = np.zeros_like(sort_idx)
        remap[sort_idx] = np.arange(len(sort_idx))
        sorted_labels = remap[cluster_ids]

        quantized = np.array([sorted_centroids[label] for label in sorted_labels]).reshape(w_sparse.shape)
        quantized_list.append(quantized)

    # Stage 2: Encode / Stage 3: Decode
    encoded = encode_weights(weights, sparsity_cfg, n_clusters)
    decoded = decode_weights(encoded)

    print("Comparing Stage 1 (quantized) to Stage 3 (decoded):")
    for i, (q, d) in enumerate(zip(quantized_list, decoded)):
        diff = np.abs(q - d)
        max_abs_err = np.max(diff)
        l2_err = np.linalg.norm(q - d)
        print(f"  Layer {i}: max(abs err)={max_abs_err:.8f}, L2={l2_err:.8f}")

    print("FedWSOcomp encode/decode symmetry test completed.")


def fedwsocomp_symmetry_test_unet():
    print("\n===== FedWSOcomp Symmetry Test (UNet) =====")
    np.random.seed(1234)
    torch.manual_seed(1234)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
        norm="batch",
    )
    state_dict = model.state_dict()
    layer_names = sorted(state_dict.keys())
    weights = [state_dict[k].cpu().numpy() for k in layer_names]

    sparsity_cfg = {"Conv": 0.5, "Linear": 0.5, "Bias": 0.2, "BatchNorm": 0.0}
    n_clusters = 16

    quantized_list = []
    for w in weights:
        if len(w.shape) in (5, 4):
            typ = "Conv"
        elif len(w.shape) == 2:
            typ = "Linear"
        else:
            typ = "Bias"

        w_sparse = fedwsocomp_top_z(w, sparsity_cfg.get(typ, 0.0))
        flat_vals = w_sparse.flatten()
        k = min(n_clusters, len(np.unique(flat_vals)))
        if k < 1:
            quantized_list.append(w_sparse)
            continue

        kmeans = KMeans(n_clusters=k, n_init=1, random_state=0)
        cluster_ids = kmeans.fit_predict(flat_vals.reshape(-1, 1))
        raw_centroids = kmeans.cluster_centers_.flatten()

        sort_idx = np.argsort(raw_centroids)
        sorted_centroids = raw_centroids[sort_idx]
        remap = np.zeros_like(sort_idx)
        remap[sort_idx] = np.arange(len(sort_idx))
        sorted_labels = remap[cluster_ids]

        quantized = np.array([sorted_centroids[label] for label in sorted_labels]).reshape(w_sparse.shape)
        quantized_list.append(quantized)

    encoded = encode_weights(weights, sparsity_cfg, n_clusters)
    decoded = decode_weights(encoded)

    print("Comparing Stage 1 (quantized) to Stage 3 (decoded):")
    all_match = True
    for i, (q, d, name) in enumerate(zip(quantized_list, decoded, layer_names)):
        if not np.allclose(q, d, atol=1e-6):
            diff = np.abs(q - d)
            max_abs_err = np.max(diff)
            l2_err = np.linalg.norm(q - d)
            print(f"  [FAIL] Layer {i} ({name}): max(abs err)={max_abs_err:.8f}, L2={l2_err:.8f}")
            all_match = False
        else:
            print(f"  [PASS] Layer {i} ({name}): Quantized and decoded match.")
    if all_match:
        print("FedWSOcomp symmetry test PASSED for UNet.")
    else:
        print("FedWSOcomp symmetry test FAILED for UNet.")


if __name__ == "__main__":
    # Optional: run tests if executed directly
    fedwsocomp_symmetry_test()
    # fedwsocomp_symmetry_test_unet()
