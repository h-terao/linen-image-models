from __future__ import annotations
from pathlib import Path
import json
import random

import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("in_dir", "data/ILSVRC2012/temp", "Input directory.", short_name="i")
flags.DEFINE_string("out_dir", "data/ILSVRC2012", "Output directory.", short_name="o")
flags.DEFINE_integer("seed", 1234, "Random seed value.", short_name="s")


def sweep_images(data_dir: str | Path):
    data_dir = Path(data_dir)

    items = []
    for class_idx, class_dir in enumerate(sorted(data_dir.iterdir())):
        for file_path in class_dir.iterdir():
            items.append([file_path, class_idx])

    random.shuffle(items)  # shuffle.
    return items


def write_tfrecord(file, items):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(file)) as writer:
        for image_file, label in tqdm(items):
            width, height = Image.open(image_file).size
            img_bytes = image_file.read_bytes()

            serialized = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                    }
                )
            ).SerializeToString()

            writer.write(serialized)


def main(argv):
    del argv  # unused.

    in_dir = Path(FLAGS.in_dir)
    out_dir = Path(FLAGS.out_dir)

    random.seed(FLAGS.seed)

    num_classes = len(list((in_dir / "train").iterdir()))
    print("NUMBER OF CLASSES:", num_classes)

    print("PROCESSING TRAIN DATA")
    train_items = sweep_images(in_dir / "train")
    write_tfrecord(out_dir / "train.tfrecord", train_items)

    print("PROCESSING VALID DATA")
    val_items = sweep_images(in_dir / "val")
    write_tfrecord(out_dir / "val.tfrecord", val_items)

    with open(out_dir / "meta.json", "w") as f:
        data_meta = {
            "num_classes": num_classes,
            "num_train": len(train_items),
            "num_val": len(val_items),
        }
        json.dump(data_meta, f)


if __name__ == "__main__":
    app.run(main)
