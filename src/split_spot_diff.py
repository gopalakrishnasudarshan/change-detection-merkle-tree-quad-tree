from pathlib import Path
from PIL import Image

INPUT_PATH = Path("data/Original_Image.jpg")  
OUT1_PATH = Path("data/Image_11.png")
OUT2_PATH = Path("data/Image_22.png")

TARGET_SIZE = (512, 512)


def main():
    img = Image.open(INPUT_PATH).convert("RGB")

    w, h = img.size
    mid = w // 2

    # Exact pixel split
    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))

    # Resize both identically
    left_resized = left.resize(TARGET_SIZE, Image.BICUBIC)
    right_resized = right.resize(TARGET_SIZE, Image.BICUBIC)

    left_resized.save(OUT1_PATH)
    right_resized.save(OUT2_PATH)

    print("Saved:")
    print(OUT1_PATH)
    print(OUT2_PATH)


if __name__ == "__main__":
    main()
