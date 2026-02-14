from pathlib import Path
from PIL import Image
import numpy as np

from src.tiling import tile_image_level
from src.hashing import patch_phash
from src.merkle import build_merkle_tree, merkle_diff_changed_leaves

IMG1_PATH = Path("data/Image_1.png")
IMG2_PATH = Path("data/Image_2.png")
OUTPUT_PATH = Path("data/output.png")

LEVEL = 4
HASH_SIZE = 8


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.array(Image.open(path).convert("RGB"))

def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    gray = np.dot(rgb[..., :3],[0.299, 0.587, 0.114])
    gray = gray.astype(np.uint8)
    return np.stack([gray,gray,gray], axis=-1)

def reconstruct_image(patches, rows, cols):
    ph, pw, ch = patches[0].shape
    out = np.zeros((rows * ph, cols * pw, ch), dtype=np.uint8)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r *ph
            y1 = y0 + ph 
            x0 = c * pw
            x1 = x0 + pw
            out[y0:y1, x0:x1] = patches[idx]
            idx += 1
    return out


def main() -> None:
    img1 = load_rgb(IMG1_PATH)
    img2 = load_rgb(IMG2_PATH)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1={img1.shape} img2={img2.shape}")

    patches1, rows, cols = tile_image_level(img1, LEVEL, pad=False)
    patches2, _, _ = tile_image_level(img2, LEVEL, pad=False)

    # Leaf hashes 
    leaf1 = [patch_phash(p, hash_size=HASH_SIZE).encode("utf-8") for p in patches1]
    leaf2 = [patch_phash(p, hash_size=HASH_SIZE).encode("utf-8") for p in patches2]

    tree1 = build_merkle_tree(leaf1)
    tree2 = build_merkle_tree(leaf2)

    root1 = tree1[-1][0]
    root2 = tree2[-1][0]
    
    changed = merkle_diff_changed_leaves(tree1, tree2)
    print("Changed patches:", changed)
    
    vis_patches = []
    for i, patch in enumerate(patches2):
        if i in changed:
            vis_patches.append(patch)  # keep color
        else:
            vis_patches.append(to_grayscale(patch))

    output_img = reconstruct_image(vis_patches, rows, cols)

    Image.fromarray(output_img).save(OUTPUT_PATH)
    print("Saved output to:", OUTPUT_PATH)


    print("Merkle roots equal?", root1 == root2)
    print("root1:", root1.hex())
    print("root2:", root2.hex())

    changed = merkle_diff_changed_leaves(tree1, tree2)
    print(f"\nMerkle changed leaves: {len(changed)}/{len(leaf1)}")
    print("changed_indices:", changed)

    if changed:
        print("\nFirst changed (index -> (row,col)):")
        for i in changed[:10]:
            r = i // cols
            c = i % cols
            print(f"{i} -> ({r},{c})")


if __name__ == "__main__":
    main()
