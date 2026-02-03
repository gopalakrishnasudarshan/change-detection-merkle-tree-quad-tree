import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal

DescriptorKind = Literal["canny","sobel_mag"]

@dataclass(frozen=True)
class DescriptorConfig: 
    kind: DescriptorKind = "canny"
    
    clip_percentiles: tuple[float, float] = (2.0, 98.0)

    eps: float = 1e-6
    
    blur_ksize: int = 3 
    
    canny_low: int = 50
    canny_high: int = 150
    
    downsample_to: Tuple[int,int] = (16,16)
    quant_step: int = 32
    

def _shared_normalize_to_uint8(t1: np.ndarray, t2: np.ndarray,
                              p_lo: float, p_hi: float, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    
    x = np.concatenate([t1.reshape(-1), t2.reshape(-1)]).astype(np.float32)
    lo = np.percentile(x,p_lo)
    hi = np.percentile(x,p_hi)
    hi = max(hi, lo + eps)
    
    def norm(tile:np.ndarray) -> np.ndarray:
         y = tile.astype(np.float32)
         y = (y - lo) / (hi - lo)
         y = np.clip(y, 0.0, 1.0)
         return (y * 255.0).round().astype(np.uint8)

    return norm(t1), norm(t2)
        
    
def _downsample_quantize(img_u8: np.ndarray, size_wh: Tuple[int, int], quant_step: int) -> np.ndarray:
    
    w, h = size_wh
    small = cv2.resize(img_u8, (w, h), interpolation = cv2.INTER_AREA)
    q = (small // quant_step) * quant_step
    return q.astype(np.uint8)


def compute_descriptor_pair(tile1: np.ndarray, tile2: np.ndarray, cfg: DescriptorConfig) -> Tuple[np.ndarray, np.ndarray]:
    t1_u8, t2_u8 = _shared_normalize_to_uint8(
        tile1, tile2,
        p_lo=cfg.clip_percentiles[0],
        p_hi=cfg.clip_percentiles[1],
        eps=cfg.eps
    )
    
    if cfg.blur_ksize and cfg.blur_ksize >=3:
         t1_u8 = cv2.GaussianBlur(t1_u8, (cfg.blur_ksize, cfg.blur_ksize), 0)
         t2_u8 = cv2.GaussianBlur(t2_u8, (cfg.blur_ksize, cfg.blur_ksize), 0)

    if cfg.kind == "canny":
        d1 = cv2.Canny(t1_u8, cfg.canny_low, cfg.canny_high)
        d2 = cv2.Canny(t2_u8, cfg.canny_low, cfg.canny_high)

    elif cfg.kind == "sobel_mag":
        def sobel_mag(x: np.ndarray) -> np.ndarray:
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag = np.clip(mag, 0, 255)
            return mag.astype(np.uint8)

        d1 = sobel_mag(t1_u8)
        d2 = sobel_mag(t2_u8)
    else:
        raise ValueError(f"Unknown descriptor kind: {cfg.kind}")

    d1q = _downsample_quantize(d1, cfg.downsample_to, cfg.quant_step)
    d2q = _downsample_quantize(d2, cfg.downsample_to, cfg.quant_step)
    return d1q, d2q