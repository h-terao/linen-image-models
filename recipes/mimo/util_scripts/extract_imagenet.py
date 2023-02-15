"""Preparing ImageNet dataset. This script extracts archives and resize images.

Usage:
    1. download `ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar` and
        `ILSVRC2012_img_val.tar` from ImageNet website and place them under `--in_dir`.
    2. run this script.

Modify from: https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py
"""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np
import scipy.io
from absl import app, flags
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


FLAGS = flags.FLAGS
flags.DEFINE_string("in_dir", None, "Input directory.", required=True, short_name="i")
flags.DEFINE_string("out_dir", "data/ILSVRC2012/temp", "Output directory.", short_name="o")
flags.DEFINE_integer(
    "size", 256, "Desired size of shorter side to resize.", lower_bound=0, short_name="s"
)
flags.DEFINE_integer("quality", 85, "JPEG quality.", lower_bound=0, short_name="q")


def _resize_worker(file_path, size: int = 256, quality: int = 85):
    im = Image.open(file_path)
    im = im.convert("RGB")

    iW, iH = im.size
    oW = round(size * iW / min(iW, iH))
    oH = round(size * iH / min(iW, iH))

    im = im.resize((oW, oH))
    im.save(file_path, quality=quality)  # overwrite.


def resize_all(out_dir, size: int = 256, quality: int = 85) -> None:
    out_dir = Path(out_dir)
    fs = list([x for x in out_dir.rglob("*") if x.is_file()])  # Sweep all file.
    for indices in tqdm(np.array_split(np.arange(len(fs)), 100), desc="==> RESIZING"):
        Parallel(n_jobs=-1)(
            delayed(_resize_worker)(fs[index], size=size, quality=quality) for index in indices
        )


def parse_devkit(in_dir: str, out_dir: str):
    shutil.unpack_archive(Path(in_dir, "ILSVRC2012_devkit_t12.tar.gz"), out_dir)
    devkit_dir = Path(out_dir, "ILSVRC2012_devkit_t12")
    try:
        # Parse meta file.
        file_path = Path(devkit_dir, "data", "meta.mat")
        meta = scipy.io.loadmat(file_path, squeeze_me=True)["synsets"]

        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]

        indices, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(x.split(", ")) for x in classes]
        idx_to_wnid = {k: v for k, v in zip(indices, wnids)}
        wnid_to_classes = {k: v for k, v in zip(wnids, classes)}

        # Parse ground_truth text.
        file_path = Path(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")
        val_indices = list(map(int, file_path.read_text().splitlines()))

        val_wnids = [idx_to_wnid[x] for x in val_indices]

    finally:
        shutil.rmtree(devkit_dir)

    return val_wnids, wnid_to_classes


def parse_train_archive(in_dir: str | Path, out_dir: str | Path, size: int, quality: int) -> None:
    tmp_dir = Path(out_dir, "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(Path(in_dir, "ILSVRC2012_img_train.tar"), "r:*") as f:
            f.extractall(tmp_dir)

        out_dir = Path(out_dir, "train")
        for x in tqdm(list((tmp_dir).iterdir()), desc="TRAIN"):
            wnid = x.stem
            (out_dir / wnid).mkdir(parents=True, exist_ok=True)
            with tarfile.open(x, "r:*") as f:
                f.extractall(out_dir / wnid)
            x.unlink()  # remove tar file.

        resize_all(out_dir, size=size, quality=quality)
    finally:
        shutil.rmtree(tmp_dir)


def parse_val_archive(
    in_dir: str | Path, out_dir: str | Path, wnids: list[str], size: int, quality: int
) -> None:
    tmp_dir = Path(out_dir, "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract.
        with tarfile.open(Path(in_dir, "ILSVRC2012_img_val.tar"), "r:*") as f:
            f.extractall(tmp_dir)

        out_dir = Path(out_dir, "val")
        for wnid in set(wnids):
            (out_dir / wnid).mkdir(parents=True)

        images = sorted(tmp_dir.iterdir())
        for wnid, src in tqdm(list(zip(wnids, images)), desc="VALID"):
            dst = Path(out_dir, wnid, src.name)
            shutil.move(src, dst)

        resize_all(out_dir, size=size, quality=quality)
    finally:
        shutil.rmtree(tmp_dir)


def main(argv):
    del argv  # unused.

    val_wnids, wnid_to_classes = parse_devkit(FLAGS.in_dir, FLAGS.out_dir)
    parse_train_archive(FLAGS.in_dir, FLAGS.out_dir, FLAGS.size, FLAGS.quality)
    parse_val_archive(FLAGS.in_dir, FLAGS.out_dir, val_wnids, FLAGS.size, FLAGS.quality)


if __name__ == "__main__":
    app.run(main)
