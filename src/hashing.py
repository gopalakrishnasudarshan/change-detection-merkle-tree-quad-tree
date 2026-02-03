import hashlib 
import numpy as np
from src.preprocess import preprocess_tile


def tile_to_bytes(tile: np.ndarray) -> bytes: 
    if tile.ndim != 2: 
        raise ValueError(f"Expected 2D tile. got shape = {tile.shape}")
    
    h,w = tile.shape
    shape_bytes = np.array([h,w], dtype= np.uint32).tobytes(order="C")

    dtype_str = str(tile.dtype).encode("ascii")
    dtype_len = np.array([len(dtype_str)], dtype=np.uint32).tobytes(order="C")
    data_bytes = np.ascontiguousarray(tile).tobytes(order="C")


    return shape_bytes + dtype_len + dtype_str + data_bytes 

def hash_tile(tile: np.ndarray,*,lo: float, hi: float, down: int = 16, q_step: int = 16) -> bytes:
    rep = preprocess_tile(tile, lo=lo, hi=hi, down=down, q_step=q_step)
    payload = tile_to_bytes(rep)
    return hashlib.sha256(payload).digest()
