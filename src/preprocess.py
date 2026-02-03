import numpy as np


def compute_norm_params(a1: np.ndarray, a2: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> tuple[float, float]:
    
    lo1, hi1 = np.percentile(a1, [p_low, p_high])
    lo2, hi2 = np.percentile(a2, [p_low, p_high])

    lo = float(min(lo1, lo2))
    hi = float(max(hi1, hi2))

    if hi <= lo:
        raise ValueError(f"Bad normalization bounds: lo={lo}, hi={hi}")

    return lo, hi


def normalize_to_uint8(tile: np.ndarray, lo: float, hi: float) -> np.ndarray:
    
    x = tile.astype(np.float32, copy=False)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)         
    x = (x * 255.0).round()
    return np.clip(x, 0, 255).astype(np.uint8)


def downsample_mean(tile: np.ndarray, out_size: int) -> np.ndarray:
    if tile.ndim != 2:
        raise ValueError(f"Expected 2D tile, got shape={tile.shape}")

    h, w = tile.shape
    if h % out_size != 0 or w % out_size != 0:
        raise ValueError(f"Tile shape {tile.shape} not divisible by out_size={out_size}")

    bh = h // out_size
    bw = w // out_size
    return tile.reshape(out_size, bh, out_size, bw).mean(axis=(1, 3))


def quantize_uint8(arr: np.ndarray, step: int) -> np.ndarray:
    
    if step <= 0:
        raise ValueError("step must be > 0")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {arr.dtype}")
    return (arr // step) * step


def preprocess_tile(tile: np.ndarray, *, lo: float, hi: float, down: int = 16, q_step: int = 16) -> np.ndarray:
  
    if tile.ndim != 2:
        raise ValueError(f"Expected 2D tile, got shape={tile.shape}")

    x8 = normalize_to_uint8(tile, lo, hi)     
    x = x8.astype(np.float32)
    x = downsample_mean(x, down)              
    x = np.clip(x, 0, 255).round().astype(np.uint8)
    x = quantize_uint8(x, q_step)            
    return x
