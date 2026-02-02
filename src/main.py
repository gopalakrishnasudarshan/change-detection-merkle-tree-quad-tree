from pathlib import Path
import hashlib
import rasterio
import numpy as np


DATA_ROOT = Path("data/OSCD")

T1_PATH = DATA_ROOT/"imgs_1_rect"/ "B04.tif"
T2_PATH = DATA_ROOT/"imgs_2_rect"/"B04.tif"

TILE_SIZE = 64

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

def tile_to_bytes(tile: np.ndarray) -> bytes: 
    if tile.ndim != 2: 
        raise ValueError(f"Expected 2D tile. got shape = {tile.shape}")
    
    h,w = tile.shape
    shape_bytes = np.array([h,w], dtype= np.uint32).tobytes(order="C")

    dtype_str = str(tile.dtype).encode("ascii")
    dtype_len = np.array([len(dtype_str)], dtype=np.uint32).tobytes(order="C")
    data_bytes = np.ascontiguousarray(tile).tobytes(order="C")


    return shape_bytes + dtype_len + dtype_str + data_bytes 

def hash_tile(tile: np.ndarray) -> bytes:

    payload = tile_to_bytes(tile)
    return hashlib.sha256(payload).digest()



def build_merkle_tree(leaf_hashes: list[bytes]) -> list[list[bytes]]:
    if not leaf_hashes:
        raise ValueError("Cannot build Merkle tree from empty leaf list")
    
    levels: list[list[bytes]] = [leaf_hashes]

    while len(levels[-1]) > 1:
        current = levels[-1]
        next_level: list[bytes] = []

        i = 0 
        while i < len(current):
            left = current[i]
            if i + 1 < len(current):
                right = current[i+1]
            else:
                right = left
        
            parent = hashlib.sha256(left + right).digest()
            next_level.append(parent)
            i += 2
    
        levels.append(next_level)

    return levels



def main() -> None:
    t1 = read_single_band_tif(T1_PATH)
    t2 = read_single_band_tif(T2_PATH)

    print("Files loaded")
    print("t1:", T1_PATH)
    print("t1:", T2_PATH)

    print("\nShapes:")
    print("t1:",t1.shape,"dtype:",t1.dtype)
    print("t1:",t2.shape,"dtype:",t2.dtype)

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch t1={t1.shape}, t2= {t2.shape}")
    

    print("\nBasic statistics:")
    print("t1:", stats(t1))
    print("t2:", stats(t2))

    tiles1, rows,cols = tile_image(t1, TILE_SIZE)
    tiles2, rows2, cols2 = tile_image(t2,TILE_SIZE)

    if(rows,cols) != (rows2,cols2):
        raise ValueError(f"Tile grid mismatch: t1 = ({rows},{cols}) t2=({rows2},{cols2})")

    print("\nTiling:")
    print(f" tile_size={TILE_SIZE}")
    print(f" grid={rows} rows x {cols} cols => total_tiles={len(tiles1)}")
    print(f" first_tile_shape={tiles1[0].shape} last_tile_shape={tiles1[-1].shape}")


    leaf_hashes1 = [hash_tile(tile) for tile in tiles1]
    leaf_hashes2 = [hash_tile(tile) for tile in tiles2]

    diff_flags = [h1 != h2 for h1, h2 in zip(leaf_hashes1, leaf_hashes2)]
    diff_count = sum(diff_flags)

    print("\n Tile hasing(SHA-256):")
    print(f"leaf_hash_count={len(leaf_hashes1)}")
    print(f"differing_leaf_hashes={diff_count}/ {len(leaf_hashes1)}")

    sample_indices = [0, len(leaf_hashes1) // 2, len(leaf_hashes1) - 1]
    print("\nSample leaf hashes (hex):")
    for i in sample_indices:
        print(f" t1 leaf[{i}]: {leaf_hashes1[i].hex()}")
        print(f" t2 leaf[{i}]: {leaf_hashes2[i].hex()}")
        print(f" equal? {leaf_hashes1[i] == leaf_hashes2[i]}\n")



    check_indices = [0, len(tiles1) //2 , len(tiles1) -1]
    print("\nTile equality sanity check(t1 vs t2):")
    for i in check_indices: 
        same = np.array_equal(tiles1[i], tiles2[i])
        print(f"tile[{i}] equal? {same}")


    tree1 = build_merkle_tree(leaf_hashes1)
    tree2 = build_merkle_tree(leaf_hashes2)

    root1 = tree1[-1][0]
    root2 = tree2[-1][0]

    print("\nMerkle tree:")
    print(f" levels_t1={len(tree1)} root_t1={root1.hex()}")
    print(f" levels_t2={len(tree2)} root_t2={root2.hex()}")
    print(f" roots_equal? {root1 == root2}")


    


    

if __name__ == "__main__":
    main()


