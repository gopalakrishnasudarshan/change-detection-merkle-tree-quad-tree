import numpy as np



def tile_image(arr: np.ndarray, tile_size: int) -> tuple[list[np.ndarray], int, int ]:

    if arr.ndim != 2: 
        raise ValueError(f"Expected 2d Array. got shape = {arr.shape}")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    
    h,w = arr.shape
    n_rows = (h + tile_size - 1) // tile_size
    n_cols = (w + tile_size -1) // tile_size

    tiles: list[np.ndarray] = []

    for r in range(n_rows):
        y0 = r * tile_size
        y1 = min(y0 + tile_size,h)

        for c in range(n_cols):
            x0 = c* tile_size
            x1 = min(x0+ tile_size,w)

            tile = arr[y0:y1, x0:x1]

            if tile.shape != (tile_size, tile_size):
                padded = np.zeros((tile_size,tile_size), dtype=arr.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append(tile)
    
    expected = n_rows * n_cols
    if len(tiles) != expected:
        raise RuntimeError(f"Tiling bug: tiles={len(tiles)} expected={expected}")
    

    return tiles, n_rows, n_cols