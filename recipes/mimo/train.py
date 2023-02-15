"""Training MIMO networks from scratch."""
from __future__ import annotations
import typing as tp
from pathlib import Path
from functools import partial
import math
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
from einops import rearrange, reduce
from flax import linen
from flax.training import dynamic_scale as dynamic_scale_lib
import limo
import optax
import alopex as ap
import chex
import tensorflow as tf
from tqdm import tqdm
from absl import app, flags

import resnet_preprocessing

tf.config.set_visible_devices([], "GPU")  # TF should not use GPUs.

flags.DEFINE_string("work_dir", "outputs", "Working directory", short_name="o")
flags.DEFINE_string("data_dir", "../data", "Dataset directory")
flags.DEFINE_integer("max_epochs", 150, "Number of training epochs.", short_name="e")
flags.DEFINE_integer("batch_size", 512, "Total batch size.", short_name="b")
flags.DEFINE_integer("seed", 0, "Random seed.", short_name="s")
flags.DEFINE_string("model_name", "resnet50", "Model name.")
flags.DEFINE_integer("ensemble_size", 2, "Ensemble size.")
flags.DEFINE_float("shuffle_rate", 0.6, "Shuffle rate.")
flags.DEFINE_integer("repetitions", 2, "Repetition of training batch.")
flags.DEFINE_float("base_learning_rate", 0.1, "Base learning rate.", short_name="lr")
flags.DEFINE_integer("warmup_epochs", 5, "Number of epochs to warmup learning rate.")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay rate.", short_name="wd")
flags.DEFINE_multi_integer("lr_decay_epochs", [30, 60, 90], "Epochs to decay learning rate.")
flags.DEFINE_bool("half_precision", False, "Whether to use mixed precision training.")

FLAGS = flags.FLAGS
IMAGENET1K_TRAIN_IMAGES = 1281167
IMAGENET1K_VALID_IMAGES = 50000
NUM_CLASSES = 1000
SHUFFLE_BUFFER_SIZE = 65536
VERSION = "V1"

Batch = tuple[chex.Array, chex.Array]


class TrainState(tp.NamedTuple):
    """A simple container to hold training state."""

    step: int
    rng: chex.PRNGKey
    params: chex.ArrayTree
    state: chex.ArrayTree
    opt_state: optax.OptState
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_data():
    def map_fun(serialized, is_training: bool = False):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(serialized, features)
        image_bytes, label = example["image"], example["label"]  # parse.

        # Decode and transform image.
        image = resnet_preprocessing.preprocess_image(image_bytes, is_training=is_training)

        # Normalize image.
        mean = tf.constant(limo.IMAGENET_DEFAULT_MEAN, dtype=tf.float32, shape=(1, 1, 3))
        std = tf.constant(limo.IMAGENET_DEFAULT_STD, dtype=tf.float32, shape=(1, 1, 3))
        image = (image - mean) / std

        # Flatten label.
        return image, label

    opts = tf.data.Options()
    opts.threading.max_intra_op_parallelism = 1

    train_path = Path(FLAGS.data_dir, "ILSVRC2012", "train.tfrecord")
    train_data = (
        tf.data.TFRecordDataset(str(train_path))
        .with_options(opts)
        .cache()
        .repeat()
        .shuffle(IMAGENET1K_TRAIN_IMAGES)
        .map(partial(map_fun, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size // FLAGS.repetitions, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    val_path = Path(FLAGS.data_dir, "ILSVRC2012", "val.tfrecord")
    val_data = (
        tf.data.TFRecordDataset(str(val_path))
        .with_options(opts)
        .map(partial(map_fun, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    return train_data, val_data


def create_model():
    return limo.create_model(
        FLAGS.model_name,
        num_classes=NUM_CLASSES * FLAGS.ensemble_size,
        dtype=jnp.float16 if FLAGS.half_precision else jnp.float32,
        norm_dtype=jnp.float32,
        axis_name="batch",
    )


def schedule_lr(step: int) -> float:
    train_batch_size = FLAGS.batch_size // FLAGS.repetitions
    steps_per_epoch = IMAGENET1K_TRAIN_IMAGES // train_batch_size
    base_lr = FLAGS.base_learning_rate * train_batch_size / 256

    epoch_details = step / steps_per_epoch
    num_decayed = jnp.sum(jnp.array(FLAGS.lr_decay_epochs) < epoch_details)
    main_lr = base_lr * jnp.power(0.1, num_decayed)
    warmup_lr = base_lr * epoch_details / FLAGS.warmup_epochs

    learning_rate = jnp.where(epoch_details < FLAGS.warmup_epochs, warmup_lr, main_lr)
    return learning_rate


def accuracy(inputs, labels, k: int = 1):
    assert inputs.shape == labels.shape
    y = jnp.argsort(inputs)[..., -k:]
    t = jnp.argmax(labels, axis=-1, keepdims=True)
    return jnp.sum(y == t, axis=-1)


def initialize(rng, batch) -> TrainState:
    init_rng, state_rng = jr.split(rng)
    inputs = batch[0][:1]  # only need a sample.

    inputs = jnp.concatenate([inputs] * FLAGS.ensemble_size, axis=-1)
    variables = create_model().init(init_rng, inputs)
    state, params = variables.pop("params")

    learning_rate = schedule_lr(step=0)
    return TrainState(
        step=0,
        rng=state_rng,
        params=params,
        state=state,
        opt_state=optax.sgd(learning_rate, momentum=0.9, nesterov=True).init(params),
        dynamic_scale=dynamic_scale_lib.DynamicScale() if FLAGS.half_precision else None,
    )


@partial(ap.train_epoch, axis_name="batch")
def train_epoch(train_state: TrainState, batch: Batch):
    # This is step function for training, but transformed to epoch function by `alopex.train_epoch`.
    print("compiling...")

    inputs, labels = batch
    labels = linen.one_hot(labels, NUM_CLASSES)

    rng, new_rng = jr.split(train_state.rng)
    rng = jr.fold_in(rng, jax.lax.axis_index("batch"))

    inputs = jnp.concatenate([inputs] * FLAGS.repetitions, axis=0)
    labels = jnp.concatenate([labels] * FLAGS.repetitions, axis=0)

    rng, shuffle_rng = jr.split(rng)
    index = jr.permutation(shuffle_rng, len(inputs))
    inputs, labels = inputs[index], labels[index]

    def shuffle(rng):
        shuffle_size = int(FLAGS.shuffle_rate * len(inputs))
        shuffle_index = jr.permutation(rng, shuffle_size)
        keep_index = jnp.arange(shuffle_size, len(inputs))
        index = jnp.concatenate([shuffle_index, keep_index])
        return inputs[index], labels[index]

    shuffle_rng, drop_rng = jr.split(rng)
    inputs, labels = zip(*[shuffle(x) for x in jr.split(shuffle_rng, FLAGS.ensemble_size)])
    inputs = jnp.concatenate(inputs, axis=-1)  # stack along channel dim.
    labels = jnp.stack(labels, axis=-2)  # [..., M, C], where M is ensemble size.

    def loss_fun(params):
        variables = {"params": params, **train_state.state}
        logits, new_state = create_model().apply(
            variables, inputs, rngs={"dropout": drop_rng}, mutable="batch_stats", is_training=True
        )

        logits = rearrange(logits, "... (M C) -> ... M C", M=FLAGS.ensemble_size)
        ce_loss = jnp.sum(optax.softmax_cross_entropy(logits, labels), axis=-1).mean()
        l2_loss = sum([jnp.sum(x**2) for x in tree_util.tree_leaves(params) if x.ndim > 1])
        loss = ce_loss + FLAGS.weight_decay * l2_loss

        metrics = {
            "train/loss": loss,
            "train/ce_loss": ce_loss,
            "train/l2_loss": l2_loss,
            "train/accuracy": accuracy(logits, labels).mean(),
        }

        return loss, (new_state, metrics)

    dynamic_scale = train_state.dynamic_scale
    if dynamic_scale:
        grad_fun = dynamic_scale.value_and_grad(loss_fun, has_aux=True, axis_name="batch")
        dynamic_scale, is_fin, aux, grads = grad_fun(train_state.params)
        new_state, metrics = aux[1]
    else:
        grads, (new_state, metrics) = jax.grad(loss_fun, has_aux=True)(train_state.params)
        grads = jax.lax.pmean(grads, "batch")

    learning_rate = schedule_lr(train_state.step)
    metrics["learning_rate"] = learning_rate

    optimizer = optax.sgd(learning_rate, momentum=0.9, nesterov=True)
    updates, new_opt_state = optimizer.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    if dynamic_scale:
        new_params, new_state, new_opt_state = tree_util.tree_map(
            partial(jnp.where, is_fin),
            (new_params, new_state, new_opt_state),
            (train_state.params, train_state.state, train_state.opt_state),
        )
        metrics["loss_scale"] = dynamic_scale.scale

    new_train_state = TrainState(
        rng=new_rng,
        step=train_state.step + 1,
        params=new_params,
        state=new_state,
        opt_state=new_opt_state,
        dynamic_scale=dynamic_scale,
    )

    return new_train_state, metrics


@partial(ap.eval_epoch, prefix="val/", axis_name="batch")
def eval_epoch(train_state: TrainState, batch: Batch):
    # This is step function for validation,
    # but transformed to epoch function by `alopex.eval_epoch`.
    inputs, labels = batch
    labels = linen.one_hot(labels, NUM_CLASSES)

    inputs = jnp.concatenate([inputs] * FLAGS.ensemble_size, axis=-1)
    variables = {"params": train_state.params, **train_state.state}
    logits = create_model().apply(variables, inputs)
    logits = reduce(logits, "... (M C) -> ... C", "sum", M=FLAGS.ensemble_size)

    return {
        "ce_loss": optax.softmax_cross_entropy(logits, labels).mean(),
        "accuracy": accuracy(logits, labels).mean(),
    }


def main(_):
    num_devices = jax.local_device_count()
    msg = f"Batch size should be divisible by number of devices {num_devices}."
    assert FLAGS.batch_size % (num_devices * FLAGS.repetitions) == 0, msg

    work_dir_path = Path(FLAGS.work_dir, FLAGS.model_name)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    # Save flags into text file.
    config_path = work_dir_path / "flags.txt"
    if not config_path.exists():
        FLAGS.append_flags_into_file(str(work_dir_path / "flags.txt"))

    train_batch_size = FLAGS.batch_size // FLAGS.repetitions
    train_steps_per_epoch = IMAGENET1K_TRAIN_IMAGES // train_batch_size
    val_steps_per_epoch = math.ceil(IMAGENET1K_VALID_IMAGES / FLAGS.batch_size)

    train_data, val_data = create_data()
    train_state = initialize(jr.PRNGKey(FLAGS.seed), next(train_data))
    logger = ap.LoggerCollection(ap.ConsoleLogger(), ap.DiskLogger(work_dir_path))

    to_save = {"train_state": train_state, "lg_state": logger.state_dict(), "best_score": -1}
    if (work_dir_path / "last_state.pkl").exists():
        print("Previous checkpoint is found. Resume training.")
        to_save = limo.load(work_dir_path / "last_state.pkl")
        logger.load_state_dict(to_save["lg_state"])

    start_epoch = int(to_save["train_state"].step) // train_steps_per_epoch
    for epoch in range(start_epoch, FLAGS.max_epochs):
        start_time = time.time()

        to_save["train_state"], summary = train_epoch(
            to_save["train_state"],
            tqdm(
                train_data,
                desc=f"Training [Epoch: {epoch}]",
                total=train_steps_per_epoch,
                leave=False,
            ),
            train_steps_per_epoch,
        )
        summary |= eval_epoch(
            to_save["train_state"],
            tqdm(
                val_data,
                desc=f"Validation [Epoch: {epoch}]",
                total=val_steps_per_epoch,
                leave=False,
            ),
            val_steps_per_epoch,
        )

        summary["elapsed_time"] = (time.time() - start_time) / 60
        logger.log_summary(summary, int(to_save["train_state"].step), epoch + 1)

        to_save["lg_state"] = logger.state_dict()
        if to_save["best_score"] <= summary["val/accuracy"]:
            to_save["best_score"] = summary["val/accuracy"]
            limo.save(work_dir_path / "best_state.pkl", to_save, exist_ok=True)
        limo.save(work_dir_path / "last_state.pkl", to_save, exist_ok=True)

    # Load best state and save its variables.
    train_state = limo.load(work_dir_path / "best_state.pkl")["train_state"]
    variables = {"params": train_state.params, **train_state.state}
    limo.save(
        work_dir_path / f"IMAGENET1K_MIMO_{FLAGS.ensemble_size}_{VERSION}", variables, exist_ok=True
    )

    print("All works done, moi moi (·x·)")


if __name__ == "__main__":
    app.run(main)
