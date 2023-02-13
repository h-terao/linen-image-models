from __future__ import annotations
import typing as tp
from pathlib import Path
from functools import partial
import math

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
import deeplake
import tensorflow as tf
from absl import app, flags

import resnet_preprocessing

tf.config.set_visible_devices([], "GPU")

flags.DEFINE_string("work_dir", None, "Working directory", short_name="o")
flags.DEFINE_integer("max_epochs", 150, "Number of training epochs.", short_name="e")
flags.DEFINE_integer("batch_size", 256, "Total batch size.", short_name="b")
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
SHUFFLE_BUFFER_SIZE = 32768
VERSION = "V1"


class TrainState(tp.NamedTuple):
    step: int
    rng: chex.PRNGKey
    params: chex.ArrayTree
    state: chex.ArrayTree
    opt_state: optax.OptState
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_data():
    def map_fun(x, is_training: bool = False):
        image, label = x["images"], x["labels"]  # check.

        # Transform image.
        image += tf.zeros(shape=(1, 1, 3), dtype=tf.uint8)  # channel must be three.
        image = tf.cast(image, tf.float32) / 255.0
        image = resnet_preprocessing.preprocess_image(image, is_training=is_training)

        # Normalize image.
        mean = tf.constant(limo.IMAGENET_DEFAULT_MEAN, dtype=tf.float32, shape=(1, 1, 3))
        std = tf.constant(limo.IMAGENET_DEFAULT_STD, dtype=tf.float32, shape=(1, 1, 3))
        image = (image - mean) / std

        # Flatten label.
        label = label[0]
        return image, label

    train_data = deeplake.load("hub://activeloop/imagenet-train")
    train_data = (
        train_data.tensorflow()
        .repeat()
        .map(partial(map_fun, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(FLAGS.batch_size // FLAGS.repetitions, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    val_data = deeplake.load("hub://activeloop/imagenet-val")
    val_data = (
        val_data.tensorflow()
        .map(partial(map_fun, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
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
    param_rng, init_rng = jr.split(rng)
    inputs = batch[0][:1]  # only need a sample.

    inputs = jnp.concatenate([inputs] * FLAGS.ensemble_size, axis=-1)
    variables = create_model().init(param_rng, inputs)
    state, params = variables.pop("params")

    learning_rate = schedule_lr(step=0)
    return TrainState(
        step=0,
        rng=init_rng,
        params=params,
        state=state,
        opt_state=optax.sgd(learning_rate, momentum=0.9, nesterov=True).init(params),
        dynamic_scale=dynamic_scale_lib.DynamicScale() if FLAGS.half_precision else None,
    )


@partial(ap.train_epoch, axis_name="batch")
def train_epoch(batch: tuple[chex.Array, chex.Array], train_state: TrainState):
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
            variables, inputs, rngs={"dropout": drop_rng}, mutable=True, is_training=True
        )

        logits = rearrange(logits, "... (M C) -> ... M C", M=FLAGS.ensemble_size)
        ce_loss = jnp.sum(optax.softmax_cross_entropy(logits, labels), axis=-1).mean()
        l2_loss = sum([jnp.sum(x**2) for x in tree_util.tree_flatten(params) if x.ndim > 1])
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
            lambda new_x, x: jnp.where(is_fin, new_x, x),
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


@partial(ap.eval_epoch, prefix="val", axis_name="batch")
def eval_epoch(batch: tuple[chex.Array, chex.Array], train_state: TrainState):
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

    train_batch_size = FLAGS.batch_size // FLAGS.repetitions
    train_steps_per_epoch = IMAGENET1K_TRAIN_IMAGES // train_batch_size
    val_steps_per_epoch = math.ceil(IMAGENET1K_VALID_IMAGES / FLAGS.batch_size)

    train_data, val_data = create_data(train_batch_size, FLAGS.batch_size)
    train_state = initialize(jr.PRNGKey(FLAGS.seed), next(train_data))
    logger = ap.LoggerCollection(ap.ConsoleLogger(), ap.DiskLogger(work_dir_path))

    to_save = {"train_state": train_state, "lg_state": logger.state_dict(), "best_score": -1}
    for epoch in range(FLAGS.max_epochs):
        to_save["train_state"], summary = train_epoch(
            to_save["train_state"], train_data, train_steps_per_epoch
        )
        summary |= eval_epoch(to_save["train_state"], val_data, val_steps_per_epoch)
        logger.log_summary(summary, int(train_state.step), epoch)

        to_save["lg_state"] = logger.state_dict()
        if to_save["best_score"] <= summary["val/accuracy"]:
            to_save["best_score"] = summary["val/accuracy"]
            limo.save(work_dir_path / "best_state.pkl", to_save)
        limo.save(work_dir_path / "last_state.pkl", to_save)

    # Load best state and save its variables.
    train_state = limo.load(work_dir_path / "best_state.pkl")["train_state"]
    variables = {"params": train_state.params, **train_state.state}
    limo.save(
        work_dir_path / f"IMAGENET1K_MIMO_{FLAGS.ensemble_size}_{VERSION}", variables, exist_ok=True
    )
    print("All works done, moi moi (·x·)")


if __name__ == "__main__":
    app.run(main)