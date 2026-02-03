import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from descriptor import DescriptorConfig, compute_descriptor_pair
from merkle import build_merkle_tree



def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


@dataclass(frozen=True)
class ChangeResult:
    changed_indices: List[int]
    leaf_hashes_t1: List[bytes]
    leaf_hashes_t2: List[bytes]


def edge_hash_tilepair(tile1: np.ndarray, tile2: np.ndarray, cfg: DescriptorConfig) -> Tuple[bytes, bytes]:
    d1, d2 = compute_descriptor_pair(tile1, tile2, cfg)
    h1 = sha256_bytes(d1.tobytes())
    h2 = sha256_bytes(d2.tobytes())
    return h1, h2


def run_edge_merkle_change_detection(
    tiles_t1: List[np.ndarray],
    tiles_t2: List[np.ndarray],
    cfg: DescriptorConfig
) -> ChangeResult:
    if len(tiles_t1) != len(tiles_t2):
        raise ValueError("tiles_t1 and tiles_t2 must have same length")

    leaf1: list[bytes] = []
    leaf2: list[bytes] = []

    for a, b in zip(tiles_t1, tiles_t2):
        h1, h2 = edge_hash_tilepair(a, b, cfg)
        leaf1.append(h1)
        leaf2.append(h2)

    mt1 = build_merkle_tree(leaf1)
    mt2 = build_merkle_tree(leaf2)

   
    changed = [i for i, (x, y) in enumerate(zip(leaf1, leaf2)) if x != y]

    return ChangeResult(
        changed_indices=changed,
        leaf_hashes_t1=leaf1,
        leaf_hashes_t2=leaf2
    )
