import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTENSIONS = [".jpg", ".png", ".bmp"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.images_dir.is_dir()
    args.output_file.parent.mkdir(exist_ok=True, parents=True)

    flist = list(args.images_dir.iterdir())

    f = h5py.File(args.output_file, "w")
    for file in tqdm(flist):
        ext = file.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"{file} does not have a supported extension. Skipping!!")
            continue
        if ext == ".jpg":
            fin = open(str(file), "rb")
            binary_data = fin.read()
            binary_data_np = np.asarray(binary_data)
            fin.close()
        else:
            print(f"JPEG Compression is applied to sample {file}")
            tmp = Image.open(file)
            tmp.save("temp.jpg", "jpeg", quality=100)
            fin = open("temp.jpg", "rb")
            binary_data = fin.read()
            binary_data_np = np.asarray(binary_data)
            fin.close()
            os.remove("temp.jpg")

        fname = file.stem
        f.create_dataset(fname, data=binary_data_np)
    f.close()


if __name__ == "__main__":
    main()
