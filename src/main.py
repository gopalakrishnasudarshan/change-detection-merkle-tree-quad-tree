from pathlib import Path
import numpy as np
from src.preprocess import compute_norm_params
from src.io_utils import read_single_band_tif, stats
from src.tiling import tile_image
from src.hashing import hash_tile
from src.merkle import build_merkle_tree, merkle_diff_changed_leaves
from src.preprocess import preprocess_tile



DATA_ROOT = Path("data/OSCD")
T1_PATH = DATA_ROOT / "imgs_1_rect" / "B04.tif"
T2_PATH = DATA_ROOT / "imgs_2_rect" / "B04.tif"

TILE_SIZE = 64
Q_STEP = 16
DOWNSAMPLE = 16



def main() -> None:
    t1 = read_single_band_tif(T1_PATH)
    t2 = read_single_band_tif(T2_PATH)

    print("Files loaded")
    print("t1:", T1_PATH)
    print("t2:", T2_PATH)

    print("\nShapes:")
    print("t1:", t1.shape, "dtype:", t1.dtype)
    print("t2:", t2.shape, "dtype:", t2.dtype)

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch t1={t1.shape}, t2={t2.shape}")

    print("\nBasic statistics:")
    print("t1:", stats(t1))
    print("t2:", stats(t2))
    
    lo, hi = compute_norm_params(t1, t2, p_low=2.0, p_high=98.0)
    print(f"\nNormalization bounds (shared): lo={lo:.2f}, hi={hi:.2f}")
    print(f"preprocess: down={DOWNSAMPLE} q_step={Q_STEP}")


    tiles1, rows, cols = tile_image(t1, TILE_SIZE)
    tiles2, rows2, cols2 = tile_image(t2, TILE_SIZE)

    if (rows, cols) != (rows2, cols2):
        raise ValueError(f"Tile grid mismatch: t1=({rows},{cols}) t2=({rows2},{cols2})")

    print("\nTiling:")
    print(f"tile_size={TILE_SIZE}")
    print(f"grid={rows} rows x {cols} cols => total_tiles={len(tiles1)}")
    print(f"first_tile_shape={tiles1[0].shape} last_tile_shape={tiles1[-1].shape}")

    leaf_hashes1 = [hash_tile(tile, lo=lo, hi=hi, down=DOWNSAMPLE, q_step=Q_STEP) for tile in tiles1]
    leaf_hashes2 = [hash_tile(tile, lo=lo, hi=hi, down=DOWNSAMPLE, q_step=Q_STEP) for tile in tiles2]
    
    rep_equal = 0
    for a, b in zip(tiles1, tiles2):
        ra = preprocess_tile(a, lo=lo, hi=hi, down=DOWNSAMPLE, q_step=Q_STEP)
        rb = preprocess_tile(b, lo=lo, hi=hi, down=DOWNSAMPLE, q_step=Q_STEP)
        rep_equal += int(np.array_equal(ra, rb))

    print(f"\nPreprocessed tile exact matches: {rep_equal}/{len(tiles1)}")

    diff_flags = [h1 != h2 for h1, h2 in zip(leaf_hashes1, leaf_hashes2)]
    diff_count = sum(diff_flags)

    print("\nTile hashing (SHA-256, robust):")
    print(f"preprocess: down={DOWNSAMPLE} q_step={Q_STEP}")
    print(f"differing_leaf_hashes={diff_count}/{len(leaf_hashes1)}")

    sample_indices = [0, len(leaf_hashes1) // 2, len(leaf_hashes1) - 1]
    print("\nSample leaf hashes (hex):")
    for i in sample_indices:
        print(f" t1 leaf[{i}]: {leaf_hashes1[i].hex()}")
        print(f" t2 leaf[{i}]: {leaf_hashes2[i].hex()}")
        print(f" equal? {leaf_hashes1[i] == leaf_hashes2[i]}\n")

    check_indices = [0, len(tiles1) // 2, len(tiles1) - 1]
    print("\nTile equality sanity check (t1 vs t2):")
    for i in check_indices:
        same = np.array_equal(tiles1[i], tiles2[i])
        print(f"tile[{i}] equal? {same}")

    tree1 = build_merkle_tree(leaf_hashes1)
    tree2 = build_merkle_tree(leaf_hashes2)

    root1 = tree1[-1][0]
    root2 = tree2[-1][0]

    print("\nMerkle tree:")
    print(f"levels_t1={len(tree1)} root_t1={root1.hex()}")
    print(f"levels_t2={len(tree2)} root_t2={root2.hex()}")
    print(f"roots_equal? {root1 == root2}")

    changed_tiles = merkle_diff_changed_leaves(tree1, tree2)
    print("\nMerkle diff:")
    print(f"changed_tiles={len(changed_tiles)} / {len(leaf_hashes1)}")
    print(f"first_10_changed_indices={changed_tiles[:10]}")


if __name__ == "__main__":
    main()
