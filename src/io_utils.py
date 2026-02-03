from pathlib import Path
import rasterio
import numpy as np 

def read_single_band_tif(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found{path}")
    
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected one band, found {src.count} in {path.name}")
        return src.read(1)


def stats(arr: np.ndarray) -> str:
    return(
        f"min = {arr.min()}"
        f"max= {arr.max()}"
        f"mean={float(arr.mean()): .2f}"
    )