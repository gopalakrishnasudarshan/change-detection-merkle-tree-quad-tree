import numpy as np
from PIL import Image
import imagehash

def patch_phash(patch: np.ndarray, hash_size: int = 8) -> str:
    
    if patch.ndim not in(2,3):
        raise ValueError(f"Expected patch 2d or 3D, got shape= {patch.shape}")

    if patch.dtype != np.uint8:
        patch = patch.astype(np.uint8)
        
    img = Image.fromarray(patch)
    h = imagehash.phash(img, hash_size=hash_size)
    return str(h)