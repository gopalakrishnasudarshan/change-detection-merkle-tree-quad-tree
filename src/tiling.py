import numpy as np



def tile_image_level (
    arr: np.ndarray,
    level: int , 
    *,
    pad: bool = False
    ) -> tuple[list[np.ndarray], int, int]:
    
    if level < 1: 
        raise ValueError("Level must be >= 1")
    
    if arr.ndim not in (2,3):
        raise ValueError(f"Expected 2d or 3d array, got shape ={arr.shape}")
    
    h,w = arr.shape[:2]
    rows = cols = 2 ** level
    
    if (h% rows != 0 ) or (w % cols != 0):
        if not pad:
            raise ValueError(
                f"Image size must be divisible by 2^L. "
                f"Got HxW={h}x{w}, 2^L={rows}. "
                f"Either use pad=True or choose a different level."
            )
        
        new_h = ((h + rows -1) // rows) * rows
        new_w = ((w + cols -1) // cols) * cols
        
        if arr.ndim == 2:
            padded = np.zeros((new_h, new_w), dtype=arr.dtype)
            padded[:h, :w] = arr
        else: 
            c = arr.shape[2]
            padded = np.zeros((new_h, new_w,c), dtype = arr.dtype)
            padded[:h, :w, :] = arr
        
        arr = paddedh,w = arr.shape[:2]
        
    ph = h // rows
    pw = w // cols
    
    patches : list[np.ndarray] = []
    for r in range(rows):
        y0 = r * ph
        y1 = y0 + ph
        for c in  range(cols):
            x0 = c * pw 
            x1 = x0 + pw
            patches.append(arr[y0:y1, x0:x1].copy())
    
    expected = rows * cols
    if len(patches) !=  expected: 
        raise RuntimeError(f"Tiling bug: patches={len(patches)} expected={expected}")
    
    return patches, rows, cols
            
        
    