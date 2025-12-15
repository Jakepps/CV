"""
Вариант 9: "Отладка и мониторинг обучения LeNet-5 с использованием современных инструментов"

Как запускать (примеры):
1) Полный прогон (baseline + анализ + сломанные + сравнение LR + профилирование + отчет):
   python lenet5_lenet_debug_tb.py

2) Только сравнение стратегий LR:
   python lenet5_lenet_debug_tb.py --mode compare_lr

3) Только "сломанные" прогоны:
   python lenet5_lenet_debug_tb.py --mode broken

4) Профилирование batch sizes:
   python lenet5_lenet_debug_tb.py --mode profile

5) CPU vs GPU benchmark (если есть GPU):
   python lenet5_lenet_debug_tb.py --mode cpu_gpu_benchmark

После запуска смотрите логи в папке ./runs и отчет ./runs/_report/report.md
"""

import os
import io
import sys
import json
import time
import math
import argparse
import shutil
import random
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras import layers

import re
import traceback
from glob import glob



# 0) Общие настройки / утилиты


def set_global_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_to_tf_image(fig: plt.Figure) -> tf.Tensor:
    """Matplotlib figure -> TF image tensor [1, H, W, 4]."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(img, 0)


def available_gpus() -> List[str]:
    gpus = tf.config.list_physical_devices("GPU")
    return [g.name for g in gpus]


def get_gpu_mem_mb() -> Optional[float]:
    """Текущая память GPU (MB), если TF это поддерживает и есть GPU."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return None
        info = tf.config.experimental.get_memory_info("GPU:0")
        return float(info["current"]) / (1024 ** 2)
    except Exception:
        return None


def human_mb(bytes_value: float) -> float:
    return float(bytes_value) / (1024 ** 2)


def pad_to_32(x: np.ndarray) -> np.ndarray:
    # MNIST 28x28 -> pad до 32x32 (как классический LeNet-5)
    return np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="constant")



# 1) Данные (MNIST) + варианты "поломки"


@dataclass
class DataConfig:
    normalize: bool = True
    # Если normalize=False, x останется в диапазоне [0..255], что создаёт проблемы обучения
    shuffle_buffer: int = 10_000


def load_mnist_data(cfg: DataConfig) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = pad_to_32(x_train).astype(np.float32)
    x_test = pad_to_32(x_test).astype(np.float32)

    if cfg.normalize:
        x_train /= 255.0
        x_test /= 255.0

    # channel dim
    x_train = np.expand_dims(x_train, -1)  # [N, 32, 32, 1]
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def make_tf_datasets(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    batch_size: int,
    shuffle_buffer: int = 10_000
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val



# 2) LeNet-5


@dataclass
class ModelConfig:
    initializer: str = "glorot_uniform"  # "glorot_uniform" (норма) или "bad_init"
    # "bad_init" реализуем как RandomNormal(stddev=1.5) для “плохой инициализации”


def build_lenet5(cfg: ModelConfig, num_classes: int = 10) -> keras.Model:
    if cfg.initializer == "bad_init":
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.5)
        bias_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    else:
        kernel_init = keras.initializers.GlorotUniform()
        bias_init = keras.initializers.Zeros()

    inputs = keras.Input(shape=(32, 32, 1), name="input")

    # Классический LeNet-5 (адаптировано под Keras)
    x = layers.Conv2D(6, kernel_size=5, activation="tanh",
                      kernel_initializer=kernel_init, bias_initializer=bias_init, name="conv1")(inputs)
    x = layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool1")(x)
    x = layers.Conv2D(16, kernel_size=5, activation="tanh",
                      kernel_initializer=kernel_init, bias_initializer=bias_init, name="conv2")(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(120, activation="tanh",
                     kernel_initializer=kernel_init, bias_initializer=bias_init, name="fc1")(x)
    x = layers.Dense(84, activation="tanh",
                     kernel_initializer=kernel_init, bias_initializer=bias_init, name="fc2")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_initializer=kernel_init, bias_initializer=bias_init, name="logits")(x)

    model = keras.Model(inputs, outputs, name="LeNet5")
    return model

def set_optimizer_lr(optimizer, lr: float) -> None:
    """
    Универсальная установка learning rate для TF/Keras (включая Keras 3).
    Работает если learning_rate — Variable, Tensor/ResourceVariable или float.
    """
    # 1) новый/правильный путь: optimizer.learning_rate
    if hasattr(optimizer, "learning_rate"):
        lr_obj = optimizer.learning_rate

        # если это Variable — используем assign
        if hasattr(lr_obj, "assign"):
            lr_obj.assign(float(lr))
            return

        # иногда set_value работает
        try:
            tf.keras.backend.set_value(lr_obj, float(lr))
            return
        except Exception:
            pass

        # если это просто float / property setter
        try:
            optimizer.learning_rate = float(lr)
            return
        except Exception:
            pass

    # 2) старый путь: optimizer.lr (может отсутствовать)
    if hasattr(optimizer, "lr"):
        lr_obj = optimizer.lr

        if hasattr(lr_obj, "assign"):
            lr_obj.assign(float(lr))
            return

        try:
            tf.keras.backend.set_value(lr_obj, float(lr))
            return
        except Exception:
            pass

        try:
            optimizer.lr = float(lr)
            return
        except Exception:
            pass

    raise RuntimeError(f"Cannot set learning rate for optimizer: {type(optimizer)}")



# 3) LR стратегии


@dataclass
class LRConfig:
    strategy: str = "fixed"  # fixed | step_decay | cyclical
    lr: float = 1e-3
    # step_decay:
    step_drop: float = 0.5
    step_epochs: int = 3
    # cyclical:
    base_lr: float = 1e-4
    max_lr: float = 5e-3
    step_size: int = 2000  # iterations (batches) half-cycle


def step_decay_schedule(epoch: int, lr0: float, drop: float, step_epochs: int) -> float:
    k = epoch // step_epochs
    return lr0 * (drop ** k)


class CyclicalLR(keras.callbacks.Callback):
    """
    Циклический LR (triangular).
    Обновляет learning_rate на каждом batch.
    """
    def __init__(self, base_lr: float, max_lr: float, step_size: int, writer: tf.summary.SummaryWriter, tag_prefix="lr"):
        super().__init__()
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.step_size = int(step_size)
        self.iteration = 0
        self.writer = writer
        self.tag_prefix = tag_prefix

    def _clr(self) -> float:
        cycle = math.floor(1 + self.iteration / (2 * self.step_size))
        x = abs(self.iteration / self.step_size - 2 * cycle + 1)
        scale = max(0.0, (1 - x))
        return self.base_lr + (self.max_lr - self.base_lr) * scale

    def on_train_batch_begin(self, batch, logs=None):
        self.iteration += 1
        lr = self._clr()

        set_optimizer_lr(self.model.optimizer, lr)

        with self.writer.as_default():
            tf.summary.scalar(f"{self.tag_prefix}/cyclical_lr", lr, step=self.iteration)



# 4) Продвинутые callbacks и телеметрия


class EpochTimer(keras.callbacks.Callback):
    def __init__(self, writer: tf.summary.SummaryWriter):
        super().__init__()
        self.writer = writer
        self.epoch_times: List[float] = []
        self._t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.perf_counter() - self._t0
        self.epoch_times.append(dt)
        with self.writer.as_default():
            tf.summary.scalar("perf/epoch_time_sec", dt, step=epoch)


class MemoryLogger(keras.callbacks.Callback):
    def __init__(self, writer: tf.summary.SummaryWriter):
        super().__init__()
        self.writer = writer
        self.proc = psutil.Process(os.getpid())
        self.rss_peak_mb = 0.0
        self.gpu_peak_mb = 0.0

    def on_epoch_end(self, epoch, logs=None):
        rss_mb = human_mb(self.proc.memory_info().rss)
        self.rss_peak_mb = max(self.rss_peak_mb, rss_mb)

        gpu_mb = get_gpu_mem_mb()
        if gpu_mb is not None:
            self.gpu_peak_mb = max(self.gpu_peak_mb, gpu_mb)

        with self.writer.as_default():
            tf.summary.scalar("perf/rss_mb", rss_mb, step=epoch)
            if gpu_mb is not None:
                tf.summary.scalar("perf/gpu_mem_mb", gpu_mb, step=epoch)


class GradientLogger(keras.callbacks.Callback):
    """
    Логирование градиентов и их норм.
    Для устойчивости считаем градиенты на фиксированном mini-batch (x_sample, y_sample).
    """
    def __init__(self, writer: tf.summary.SummaryWriter, sample_batch: Tuple[np.ndarray, np.ndarray], every_n_epochs: int = 1):
        super().__init__()
        self.writer = writer
        self.xs, self.ys = sample_batch
        self.every_n_epochs = int(every_n_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if self.every_n_epochs <= 0:
            return
        if epoch % self.every_n_epochs != 0:
            return

        xs = tf.convert_to_tensor(self.xs)
        ys = tf.convert_to_tensor(self.ys)

        with tf.GradientTape() as tape:
            preds = self.model(xs, training=True)
            loss = self.model.compiled_loss(ys, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)

        # Глобальные нормы (диагностика exploding/vanishing)
        grad_norms = []
        weight_norms = []
        for g, w in zip(grads, self.model.trainable_variables):
            if g is None:
                continue
            grad_norms.append(tf.norm(g))
            weight_norms.append(tf.norm(w))

        with self.writer.as_default():
            if grad_norms:
                tf.summary.scalar("diagnostics/grad_global_norm", tf.norm(tf.stack(grad_norms)), step=epoch)
            if weight_norms:
                tf.summary.scalar("diagnostics/weight_global_norm", tf.norm(tf.stack(weight_norms)), step=epoch)

            # Гистограммы по слоям/параметрам
            for g, w in zip(grads, self.model.trainable_variables):
                if g is None:
                    continue
                name = w.name.replace(":", "_")
                tf.summary.histogram(f"grads/{name}", g, step=epoch)
                tf.summary.histogram(f"weights/{name}", w, step=epoch)


class WeightChangeLogger(keras.callbacks.Callback):
    """
    Отслеживание изменения весов (дельта L2) между эпохами.
    """
    def __init__(self, writer: tf.summary.SummaryWriter):
        super().__init__()
        self.writer = writer
        self.prev_weights: Optional[List[np.ndarray]] = None
        self.delta_per_epoch: List[float] = []

    def on_train_begin(self, logs=None):
        self.prev_weights = [w.numpy().copy() for w in self.model.trainable_variables]

    def on_epoch_end(self, epoch, logs=None):
        cur = [w.numpy() for w in self.model.trainable_variables]
        deltas = []
        for pw, cw in zip(self.prev_weights, cur):
            deltas.append(np.linalg.norm((cw - pw).ravel(), ord=2))
        delta_total = float(np.linalg.norm(np.array(deltas), ord=2))
        self.delta_per_epoch.append(delta_total)
        self.prev_weights = [c.copy() for c in cur]

        with self.writer.as_default():
            tf.summary.scalar("dynamics/weight_delta_l2", delta_total, step=epoch)
            # По желанию — можно добавить per-layer, но это сильно множит теги


class ActivationLogger(keras.callbacks.Callback):
    """
    Гистограммы активаций по слоям на фиксированном batch из val.
    """
    def __init__(self, writer: tf.summary.SummaryWriter, sample_x: np.ndarray, every_n_epochs: int = 1):
        super().__init__()
        self.writer = writer
        self.sample_x = sample_x
        self.every_n_epochs = int(every_n_epochs)
        self.act_model = None
        self.layer_names: List[str] = []

    def on_train_begin(self, logs=None):
        # Берем “интересные” слои
        outputs = []
        names = []
        for layer in self.model.layers:
            if isinstance(layer, (layers.Conv2D, layers.Dense, layers.AveragePooling2D)):
                outputs.append(layer.output)
                names.append(layer.name)
        self.layer_names = names
        self.act_model = keras.Model(self.model.input, outputs, name="ActivationProbe")

    def on_epoch_end(self, epoch, logs=None):
        if self.every_n_epochs <= 0:
            return
        if epoch % self.every_n_epochs != 0:
            return
        x = tf.convert_to_tensor(self.sample_x)
        acts = self.act_model(x, training=False)
        if not isinstance(acts, (list, tuple)):
            acts = [acts]

        with self.writer.as_default():
            for name, a in zip(self.layer_names, acts):
                tf.summary.histogram(f"activations/{name}", a, step=epoch)


class ConfusionMatrixLogger(keras.callbacks.Callback):
    def __init__(self, writer: tf.summary.SummaryWriter, x_val: np.ndarray, y_val: np.ndarray, every_n_epochs: int = 1, class_names=None):
        super().__init__()
        self.writer = writer
        self.x_val = x_val
        self.y_val = y_val
        self.every_n_epochs = int(every_n_epochs)
        self.class_names = class_names or [str(i) for i in range(10)]

    def on_epoch_end(self, epoch, logs=None):
        if self.every_n_epochs <= 0:
            return
        if epoch % self.every_n_epochs != 0:
            return

        preds = self.model.predict(self.x_val, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        cm = confusion_matrix(self.y_val, y_pred)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        thresh = cm.max() * 0.6 if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        img = fig_to_tf_image(fig)
        with self.writer.as_default():
            tf.summary.image("viz/confusion_matrix", img, step=epoch)


class PredictionExamplesLogger(keras.callbacks.Callback):
    def __init__(self, writer: tf.summary.SummaryWriter, x_val: np.ndarray, y_val: np.ndarray, every_n_epochs: int = 1, max_items: int = 16):
        super().__init__()
        self.writer = writer
        self.x_val = x_val
        self.y_val = y_val
        self.every_n_epochs = int(every_n_epochs)
        self.max_items = int(max_items)

    def on_epoch_end(self, epoch, logs=None):
        if self.every_n_epochs <= 0:
            return
        if epoch % self.every_n_epochs != 0:
            return

        n = min(self.max_items, len(self.x_val))
        xs = self.x_val[:n]
        ys = self.y_val[:n]
        preds = self.model.predict(xs, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        cols = int(math.sqrt(n))
        rows = int(math.ceil(n / cols))
        fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))

        for i in range(n):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(xs[i].squeeze(), cmap="gray")
            ax.set_title(f"T:{ys[i]} P:{y_pred[i]}")
            ax.axis("off")

        img = fig_to_tf_image(fig)
        with self.writer.as_default():
            tf.summary.image("viz/val_examples_predictions", img, step=epoch)


class CustomEarlyStoppingPlateau(keras.callbacks.Callback):
    """
    Custom early stopping при плато: если val_loss не улучшается patience эпох.
    """
    def __init__(self, monitor: str = "val_loss", patience: int = 3, min_delta: float = 1e-4, writer: Optional[tf.summary.SummaryWriter] = None):
        super().__init__()
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.writer = writer
        self.best = np.inf
        self.wait = 0
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = float(logs.get(self.monitor, np.inf))
        if current < (self.best - self.min_delta):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.writer is not None:
                    with self.writer.as_default():
                        tf.summary.scalar("diagnostics/early_stop_triggered", 1.0, step=epoch)

    def on_train_end(self, logs=None):
        # можно дописать печать, но оставляем “тихо”, чтобы не мешать логам
        pass


def log_model_architecture_image(model: keras.Model, writer: tf.summary.SummaryWriter, out_path: str) -> None:
    """
    Визуализация архитектуры:
    1) TF Graph: через TensorBoard callback write_graph=True
    2) Изображение схемы: tf.keras.utils.plot_model -> картинка в TensorBoard
    """
    try:
        keras.utils.plot_model(model, to_file=out_path, show_shapes=True, expand_nested=False, dpi=150)
        # log image to TB
        img = tf.io.read_file(out_path)
        img = tf.image.decode_png(img, channels=4)
        img = tf.expand_dims(img, 0)
        with writer.as_default():
            tf.summary.image("model/architecture", img, step=0)
    except Exception as e:
        # plot_model требует pydot/graphviz в некоторых окружениях
        with writer.as_default():
            tf.summary.text("model/architecture_error", tf.constant(str(e)), step=0)



# 5) Эксперимент / тренировка


@dataclass
class TrainConfig:
    run_name: str = "baseline"
    seed: int = 42
    epochs: int = 8
    batch_size: int = 128
    val_split: float = 0.1

    # диагностика/логирование
    tb_histogram_freq: int = 1
    profile_batch: int = 0  # 0 = выключено; можно поставить, например, 10 для профиля 10-го batch
    activation_every_n_epochs: int = 1
    grad_every_n_epochs: int = 1
    cm_every_n_epochs: int = 1
    examples_every_n_epochs: int = 1

    # оптимизация/стабильность
    optimizer: str = "adam"  # adam | sgd
    high_lr_broken: bool = False  # для “сломанных” запусков
    terminate_on_nan: bool = True

    # callbacks: ReduceLROnPlateau / checkpoint / early stop
    use_reduce_on_plateau: bool = True
    reduce_factor: float = 0.5
    reduce_patience: int = 2
    reduce_min_lr: float = 1e-6

    use_checkpoint: bool = True
    checkpoint_monitor: str = "val_accuracy"

    use_custom_early_stop: bool = True
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4

    # device label (для отчетов)
    device_label: str = "auto"  # auto|cpu|gpu (внутри скрипта — для репортинга)


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    lr: LRConfig
    train: TrainConfig


def compile_model(model: keras.Model, train_cfg: TrainConfig, lr_cfg: LRConfig) -> keras.Model:
    if train_cfg.optimizer == "sgd":
        lr = lr_cfg.lr
        if train_cfg.high_lr_broken:
            lr = max(lr, 0.5)  # “слишком высокий LR”
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        lr = lr_cfg.lr
        if train_cfg.high_lr_broken:
            lr = max(lr, 0.05)  # для Adam это уже очень агрессивно
        opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_experiment(cfg: ExperimentConfig, root_dir: str = "runs") -> Dict[str, Any]:
    set_global_determinism(cfg.train.seed)

    run_dir = os.path.join(root_dir, f"{cfg.train.run_name}_{now_tag()}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "artifacts"))

    # 1) данные
    (x_all, y_all), (x_test, y_test) = load_mnist_data(cfg.data)

    # train/val split
    n = len(x_all)
    val_n = int(n * cfg.train.val_split)
    x_val, y_val = x_all[:val_n], y_all[:val_n]
    x_train, y_train = x_all[val_n:], y_all[val_n:]

    ds_train, ds_val = make_tf_datasets(
        x_train, y_train, x_val, y_val,
        batch_size=cfg.train.batch_size,
        shuffle_buffer=cfg.data.shuffle_buffer
    )

    # sample batches for gradients/activations
    # (фиксируем небольшой batch, чтобы телеметрия была стабильной и не слишком дорогой)
    xg = x_train[:256]
    yg = y_train[:256]

    xa = x_val[:256]

    # 2) модель
    model = build_lenet5(cfg.model)
    model = compile_model(model, cfg.train, cfg.lr)

    # Writers
    writer = tf.summary.create_file_writer(run_dir)

    # 3) архитектура в TensorBoard (image + graph)
    arch_png = os.path.join(run_dir, "artifacts", "model_arch.png")
    log_model_architecture_image(model, writer, arch_png)

    # 4) callbacks
    callbacks: List[keras.callbacks.Callback] = []

    tb_cb = keras.callbacks.TensorBoard(
        log_dir=run_dir,
        histogram_freq=cfg.train.tb_histogram_freq,  # weights distributions
        write_graph=True,
        update_freq="epoch",
        profile_batch=cfg.train.profile_batch
    )
    callbacks.append(tb_cb)

    # расширенная телеметрия
    timer_cb = EpochTimer(writer)
    mem_cb = MemoryLogger(writer)
    wchg_cb = WeightChangeLogger(writer)
    grad_cb = GradientLogger(writer, sample_batch=(xg, yg), every_n_epochs=cfg.train.grad_every_n_epochs)
    act_cb = ActivationLogger(writer, sample_x=xa, every_n_epochs=cfg.train.activation_every_n_epochs)
    cm_cb = ConfusionMatrixLogger(writer, x_val=x_val, y_val=y_val, every_n_epochs=cfg.train.cm_every_n_epochs)
    ex_cb = PredictionExamplesLogger(writer, x_val=x_val, y_val=y_val, every_n_epochs=cfg.train.examples_every_n_epochs)

    callbacks += [timer_cb, mem_cb, wchg_cb, grad_cb, act_cb, cm_cb, ex_cb]

    # advanced callbacks
    if cfg.train.use_reduce_on_plateau:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.train.reduce_factor,
                patience=cfg.train.reduce_patience,
                min_lr=cfg.train.reduce_min_lr,
                verbose=1
            )
        )

    best_path = os.path.join(run_dir, "artifacts", "best.weights.h5")
    if cfg.train.use_checkpoint:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=best_path,
                monitor=cfg.train.checkpoint_monitor,
                mode="max" if "acc" in cfg.train.checkpoint_monitor else "min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        )

    if cfg.train.use_custom_early_stop:
        callbacks.append(
            CustomEarlyStoppingPlateau(
                monitor="val_loss",
                patience=cfg.train.early_stop_patience,
                min_delta=cfg.train.early_stop_min_delta,
                writer=writer
            )
        )

    if cfg.train.terminate_on_nan:
        callbacks.append(keras.callbacks.TerminateOnNaN())

    # LR schedules
    if cfg.lr.strategy == "step_decay":
        callbacks.append(
            keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: step_decay_schedule(epoch, cfg.lr.lr, cfg.lr.step_drop, cfg.lr.step_epochs),
                verbose=1
            )
        )

    if cfg.lr.strategy == "cyclical":
        callbacks.append(
            CyclicalLR(
                base_lr=cfg.lr.base_lr,
                max_lr=cfg.lr.max_lr,
                step_size=cfg.lr.step_size,
                writer=writer
            )
        )

    # 5) тренировка
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        verbose=2
    )

    # 6) загрузить лучшие веса, если есть
    best_loaded = False
    if cfg.train.use_checkpoint and os.path.exists(best_path):
        model.load_weights(best_path)
        best_loaded = True

    # 7) оценка на test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # 8) корреляция: изменения весов vs улучшение accuracy
    # берём (weight_delta_l2 per epoch) и val_accuracy
    deltas = wchg_cb.delta_per_epoch
    val_accs = history.history.get("val_accuracy", [])
    corr = None
    if len(deltas) == len(val_accs) and len(deltas) >= 2:
        corr = float(np.corrcoef(np.array(deltas), np.array(val_accs))[0, 1])
        with writer.as_default():
            tf.summary.scalar("analysis/corr_weight_delta_vs_val_acc", corr, step=len(deltas))

        # scatter plot
        fig = plt.figure(figsize=(5.5, 4.5))
        plt.scatter(deltas, val_accs)
        plt.xlabel("Weight delta L2 (epoch)")
        plt.ylabel("Validation accuracy")
        plt.title("Weight change vs val accuracy")
        img = fig_to_tf_image(fig)
        with writer.as_default():
            tf.summary.image("analysis/weight_delta_vs_val_acc_scatter", img, step=len(deltas))

    # 9) summary + артефакты
    best_val_acc = float(np.max(history.history.get("val_accuracy", [0.0])))
    best_val_loss = float(np.min(history.history.get("val_loss", [np.inf])))
    epochs_trained = len(history.history.get("loss", []))
    avg_epoch_time = float(np.mean(timer_cb.epoch_times)) if timer_cb.epoch_times else None

    summary = {
        "run_dir": run_dir,
        "run_name": cfg.train.run_name,
        "device_label": cfg.train.device_label,
        "data": asdict(cfg.data),
        "model": asdict(cfg.model),
        "lr": asdict(cfg.lr),
        "train": asdict(cfg.train),
        "epochs_trained": epochs_trained,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "avg_epoch_time_sec": avg_epoch_time,
        "rss_peak_mb": float(mem_cb.rss_peak_mb),
        "gpu_peak_mb": float(mem_cb.gpu_peak_mb) if mem_cb.gpu_peak_mb > 0 else None,
        "corr_weight_delta_vs_val_acc": corr,
        "best_weights_loaded": best_loaded,
        "gpus_visible": available_gpus(),
    }

    save_json(os.path.join(run_dir, "summary.json"), summary)
    # история в CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    # шаблон для “быстрого развёртывания похожих экспериментов”
    template = {
        "data": asdict(cfg.data),
        "model": asdict(cfg.model),
        "lr": asdict(cfg.lr),
        "train": {k: v for k, v in asdict(cfg.train).items() if k not in ("run_name",)}
    }
    save_json(os.path.join(run_dir, "experiment_template.json"), template)

    return summary



# 6) Набор “сломанных” экспериментов


def broken_suite(root_dir="runs") -> List[Dict[str, Any]]:
    results = []

    # A) слишком высокий LR
    cfg_high_lr = ExperimentConfig(
        data=DataConfig(normalize=True),
        model=ModelConfig(initializer="glorot_uniform"),
        lr=LRConfig(strategy="fixed", lr=1e-3),
        train=TrainConfig(run_name="broken_high_lr", epochs=6, batch_size=128, optimizer="adam", high_lr_broken=True)
    )
    results.append(run_experiment(cfg_high_lr, root_dir=root_dir))

    # B) плохая инициализация
    cfg_bad_init = ExperimentConfig(
        data=DataConfig(normalize=True),
        model=ModelConfig(initializer="bad_init"),
        lr=LRConfig(strategy="fixed", lr=1e-3),
        train=TrainConfig(run_name="broken_bad_init", epochs=6, batch_size=128, optimizer="adam")
    )
    results.append(run_experiment(cfg_bad_init, root_dir=root_dir))

    # C) отсутствие нормализации
    cfg_no_norm = ExperimentConfig(
        data=DataConfig(normalize=False),  # ключевая “ошибка”
        model=ModelConfig(initializer="glorot_uniform"),
        lr=LRConfig(strategy="fixed", lr=1e-3),
        train=TrainConfig(run_name="broken_no_norm", epochs=6, batch_size=128, optimizer="adam")
    )
    results.append(run_experiment(cfg_no_norm, root_dir=root_dir))

    return results



# 7) Сравнение стратегий обучения


def compare_lr_strategies(root_dir="runs") -> List[Dict[str, Any]]:
    results = []

    base_data = DataConfig(normalize=True)
    base_model = ModelConfig(initializer="glorot_uniform")
    base_train = TrainConfig(run_name="compare_fixed", epochs=8, batch_size=128, optimizer="adam")

    # 1) fixed
    cfg_fixed = ExperimentConfig(
        data=base_data,
        model=base_model,
        lr=LRConfig(strategy="fixed", lr=1e-3),
        train=base_train
    )
    results.append(run_experiment(cfg_fixed, root_dir=root_dir))

    # 2) step decay
    cfg_step = ExperimentConfig(
        data=base_data,
        model=base_model,
        lr=LRConfig(strategy="step_decay", lr=1e-3, step_drop=0.5, step_epochs=3),
        train=TrainConfig(run_name="compare_step_decay", epochs=10, batch_size=128, optimizer="adam")
    )
    results.append(run_experiment(cfg_step, root_dir=root_dir))

    # 3) cyclical
    cfg_cyc = ExperimentConfig(
        data=base_data,
        model=base_model,
        lr=LRConfig(strategy="cyclical", lr=1e-3, base_lr=1e-4, max_lr=5e-3, step_size=2000),
        train=TrainConfig(run_name="compare_cyclical", epochs=8, batch_size=128, optimizer="adam")
    )
    results.append(run_experiment(cfg_cyc, root_dir=root_dir))

    return results



# 8) Профилирование batch sizes + память


def profile_batch_sizes(root_dir="runs", batch_sizes=(32, 64, 128, 256), epochs=2) -> Dict[str, Any]:
    prof_dir = os.path.join(root_dir, f"profile_batch_{now_tag()}")
    ensure_dir(prof_dir)

    rows = []
    for bs in batch_sizes:
        cfg = ExperimentConfig(
            data=DataConfig(normalize=True),
            model=ModelConfig(initializer="glorot_uniform"),
            lr=LRConfig(strategy="fixed", lr=1e-3),
            train=TrainConfig(
                run_name=f"profile_bs{bs}",
                epochs=epochs,
                batch_size=bs,
                optimizer="adam",
                # профилирование логов “облегчаем”, чтобы измерения были чище
                tb_histogram_freq=0,
                activation_every_n_epochs=0,
                grad_every_n_epochs=0,
                cm_every_n_epochs=0,
                examples_every_n_epochs=0,
                use_reduce_on_plateau=False,
                use_checkpoint=False,
                use_custom_early_stop=False,
                device_label="auto"
            )
        )
        summary = run_experiment(cfg, root_dir=root_dir)
        rows.append({
            "batch_size": bs,
            "avg_epoch_time_sec": summary["avg_epoch_time_sec"],
            "rss_peak_mb": summary["rss_peak_mb"],
            "gpu_peak_mb": summary["gpu_peak_mb"],
            "test_accuracy": summary["test_accuracy"],
        })

    df = pd.DataFrame(rows).sort_values("batch_size")
    df.to_csv(os.path.join(prof_dir, "batch_profile.csv"), index=False)
    save_json(os.path.join(prof_dir, "batch_profile.json"), rows)

    return {"profile_dir": prof_dir, "table": rows}



# 9) CPU vs GPU (через self-subprocess)


def self_subprocess_run(device: str, root_dir="runs") -> Dict[str, Any]:
    """
    Чтобы корректно сравнить CPU vs GPU, нужно запускать отдельным процессом,
    т.к. видимость GPU фиксируется при старте TF.
    """
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # выключаем GPU для процесса
    # иначе оставляем как есть

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "single_benchmark",
        "--device", device,
        "--root", root_dir
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Subprocess failed ({device}). stderr:\n{p.stderr}")

    # subprocess печатает путь к summary.json в stdout последней строкой
    last_line = p.stdout.strip().splitlines()[-1].strip()
    if not last_line.endswith("summary.json") or not os.path.exists(last_line):
        raise RuntimeError(f"Cannot find summary.json from subprocess output: {last_line}")

    return load_json(last_line)


def cpu_gpu_benchmark(root_dir="runs") -> Dict[str, Any]:
    gpus = available_gpus()
    cpu_sum = self_subprocess_run("cpu", root_dir=root_dir)
    gpu_sum = None
    if gpus:
        gpu_sum = self_subprocess_run("gpu", root_dir=root_dir)

    out_dir = os.path.join(root_dir, f"cpu_gpu_benchmark_{now_tag()}")
    ensure_dir(out_dir)

    save_json(os.path.join(out_dir, "cpu_summary.json"), cpu_sum)
    if gpu_sum is not None:
        save_json(os.path.join(out_dir, "gpu_summary.json"), gpu_sum)

    result = {
        "benchmark_dir": out_dir,
        "cpu": cpu_sum,
        "gpu": gpu_sum,
        "gpus_visible": gpus
    }
    save_json(os.path.join(out_dir, "benchmark.json"), result)
    return result


def single_benchmark(device: str, root_dir="runs") -> str:
    # Минималистичная тренировка для измерения скорости
    cfg = ExperimentConfig(
        data=DataConfig(normalize=True),
        model=ModelConfig(initializer="glorot_uniform"),
        lr=LRConfig(strategy="fixed", lr=1e-3),
        train=TrainConfig(
            run_name=f"bench_{device}",
            epochs=3,
            batch_size=256,
            optimizer="adam",
            tb_histogram_freq=0,
            activation_every_n_epochs=0,
            grad_every_n_epochs=0,
            cm_every_n_epochs=0,
            examples_every_n_epochs=0,
            use_reduce_on_plateau=False,
            use_checkpoint=False,
            use_custom_early_stop=False,
            device_label=device
        )
    )
    summary = run_experiment(cfg, root_dir=root_dir)
    summary_path = os.path.join(summary["run_dir"], "summary.json")
    # Важно: печатаем путь для родительского процесса
    print(summary_path)
    return summary_path



# 10) Отчет (Markdown) + рекомендации


def summarize_runs(root_dir="runs") -> pd.DataFrame:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "summary.json" in filenames:
            try:
                s = load_json(os.path.join(dirpath, "summary.json"))
                rows.append({
                    "run_name": s.get("run_name"),
                    "run_dir": s.get("run_dir"),
                    "lr_strategy": s.get("lr", {}).get("strategy"),
                    "lr": s.get("lr", {}).get("lr"),
                    "epochs_trained": s.get("epochs_trained"),
                    "best_val_accuracy": s.get("best_val_accuracy"),
                    "test_accuracy": s.get("test_accuracy"),
                    "avg_epoch_time_sec": s.get("avg_epoch_time_sec"),
                    "rss_peak_mb": s.get("rss_peak_mb"),
                    "gpu_peak_mb": s.get("gpu_peak_mb"),
                    "corr_weight_delta_vs_val_acc": s.get("corr_weight_delta_vs_val_acc"),
                    "normalize": s.get("data", {}).get("normalize"),
                    "initializer": s.get("model", {}).get("initializer"),
                })
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["run_name", "best_val_accuracy"], ascending=[True, False])


def generate_recommendations(df: pd.DataFrame) -> List[str]:
    recs = []
    if df.empty:
        return ["Недостаточно данных: не найдено summary.json в runs/. Запустите эксперименты."]

    # Базовые эвристики
    broken = df[df["run_name"].astype(str).str.startswith("broken_")]
    if not broken.empty:
        # отсутствие нормализации
        no_norm = broken[broken["normalize"] == False]
        if not no_norm.empty:
            recs.append("Всегда нормализуйте входные данные (например, MNIST: деление на 255). Без нормализации ухудшается сходимость и стабильность градиентов.")

        # bad init
        bad_init = broken[broken["initializer"] == "bad_init"]
        if not bad_init.empty:
            recs.append("Используйте устойчивые инициализации (Glorot/He). Слишком “широкая” RandomNormal может провоцировать нестабильные активации/градиенты и замедлять обучение.")

        # high LR
        high_lr = broken[broken["run_name"].astype(str).str.contains("high_lr")]
        if not high_lr.empty:
            recs.append("Слишком высокий learning rate часто приводит к колебаниям loss/accuracy или к NaN. Начинайте с 1e-3 (Adam) и включайте ReduceLROnPlateau / scheduler.")

    # Лучший LR strategy среди compare_*
    compare = df[df["run_name"].astype(str).str.startswith("compare_")]
    if not compare.empty:
        best = compare.sort_values("test_accuracy", ascending=False).iloc[0]
        recs.append(f"По результатам сравнения, наиболее эффективная стратегия (test_accuracy максимум): {best['run_name']} (strategy={best['lr_strategy']}). Рекомендуется использовать её как дефолт для похожих задач.")
        recs.append("Для ускорения выхода на плато используйте: ReduceLROnPlateau + сохранение best weights (ModelCheckpoint) + early stopping.")

    # Корреляция динамики весов и точности
    corr = df["corr_weight_delta_vs_val_acc"].dropna()
    if len(corr) > 0:
        c = float(corr.mean())
        recs.append(f"Средняя корреляция между изменением весов и val_accuracy (по доступным прогонам): {c:.3f}. Используйте этот сигнал как индикатор: если веса почти не меняются при низкой точности — вероятно vanishing gradients/слишком маленький LR.")

    return recs

def _register_cyrillic_font_or_fallback() -> str:
    """
    ReportLab по умолчанию может некорректно отображать кириллицу на некоторых системах.
    Эта функция пытается подключить шрифт с кириллицей (DejaVuSans/Arial).
    Возвращает имя шрифта для использования в стилях.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return "Helvetica"

    candidates = [
        # Windows
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\times.ttf",
        # Linux
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        r"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS (на всякий случай)
        r"/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        r"/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                font_name = "ReportFontCyr"
                pdfmetrics.registerFont(TTFont(font_name, path))
                return font_name
            except Exception:
                continue

    # fallback (может не поддерживать кириллицу в некоторых окружениях)
    return "Helvetica"


def _escape_paragraph_text(s: str) -> str:
    # для Paragraph (ReportLab использует mini-HTML)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s


_inline_code_re = re.compile(r"`([^`]+)`")


def _format_inline_code(s: str) -> str:
    """
    Заменяет `code` на моноширинный шрифт внутри Paragraph.
    """
    s = _escape_paragraph_text(s)

    def repl(m):
        code = _escape_paragraph_text(m.group(1))
        return f'<font face="Courier">{code}</font>'

    return _inline_code_re.sub(repl, s)


def render_pdf_from_report_md(
    report_md_path: str,
    pdf_path: str,
    summary_csv_path: Optional[str] = None,
    images: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    report.md -> report.pdf + (опционально) таблица CSV + (опционально) изображения.
    images: список (caption, image_path).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Preformatted,
        ListFlowable, ListItem, Table, TableStyle, PageBreak
    )
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader

    ensure_dir(os.path.dirname(pdf_path))

    # --- font ---
    font_name = _register_cyrillic_font_or_fallback()
    styles = getSampleStyleSheet()

    base = ParagraphStyle(
        "BodyCyr",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )
    h1 = ParagraphStyle("H1Cyr", parent=styles["Heading1"], fontName=font_name, spaceAfter=10)
    h2 = ParagraphStyle("H2Cyr", parent=styles["Heading2"], fontName=font_name, spaceAfter=8)
    h3 = ParagraphStyle("H3Cyr", parent=styles["Heading3"], fontName=font_name, spaceAfter=6)

    caption_style = ParagraphStyle(
        "CaptionCyr",
        parent=base,
        fontSize=10,
        leading=12,
        textColor=colors.grey,
        spaceAfter=10
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        backColor=colors.whitesmoke,
        borderColor=colors.lightgrey,
        borderWidth=0.5,
        borderPadding=6,
        leftIndent=6,
        rightIndent=6,
        spaceAfter=10,
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="LeNet-5 + TensorBoard report",
        author="Generated by script",
    )

    with open(report_md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    story = []
    story.append(Paragraph(_format_inline_code("Отчёт сгенерирован автоматически на основе `report.md`."), base))
    story.append(Paragraph(_format_inline_code(f"Источник: `{os.path.basename(report_md_path)}`"), base))
    story.append(Paragraph(_format_inline_code(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), base))
    story.append(Spacer(1, 10))

    in_code = False
    code_buf: List[str] = []
    bullet_buf: List[str] = []

    def flush_bullets():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        items = [ListItem(Paragraph(_format_inline_code(b), base)) for b in bullet_buf]
        story.append(ListFlowable(items, bulletType="bullet", leftIndent=14))
        story.append(Spacer(1, 4))
        bullet_buf = []

    def flush_code():
        nonlocal code_buf
        if not code_buf:
            return
        story.append(Preformatted("\n".join(code_buf), code_style))
        code_buf = []

    for raw in lines:
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                flush_code()
            else:
                flush_bullets()
                in_code = True
                code_buf = []
            continue

        if in_code:
            code_buf.append(line)
            continue

        if line.strip() == "":
            flush_bullets()
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(Paragraph(_format_inline_code(line[2:].strip()), h1))
            continue
        if line.startswith("## "):
            flush_bullets()
            story.append(Paragraph(_format_inline_code(line[3:].strip()), h2))
            continue
        if line.startswith("### "):
            flush_bullets()
            story.append(Paragraph(_format_inline_code(line[4:].strip()), h3))
            continue

        if line.lstrip().startswith("- "):
            bullet_buf.append(line.lstrip()[2:].strip())
            continue

        flush_bullets()
        story.append(Paragraph(_format_inline_code(line), base))

    flush_bullets()
    if in_code:
        flush_code()

    # --- CSV table (опционально) ---
    if summary_csv_path and os.path.exists(summary_csv_path):
        story.append(PageBreak())
        story.append(Paragraph("Сводная таблица запусков (runs_summary.csv)", h2))
        try:
            df = pd.read_csv(summary_csv_path)
            if not df.empty:
                df2 = df.head(25).copy()
                preferred = [
                    "run_name", "lr_strategy", "lr", "epochs_trained",
                    "best_val_accuracy", "test_accuracy", "avg_epoch_time_sec", "rss_peak_mb"
                ]
                cols = [c for c in preferred if c in df2.columns]
                if cols:
                    df2 = df2[cols]
                data = [list(df2.columns)] + df2.values.tolist()

                tbl = Table(data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(Spacer(1, 10))
                story.append(tbl)
        except Exception as e:
            story.append(Spacer(1, 8))
            story.append(Paragraph(_format_inline_code(f"Не удалось вставить таблицу CSV: {e}"), base))

    # --- Images (опционально) ---
    if images:
        story.append(PageBreak())
        story.append(Paragraph("Визуальные материалы", h2))

        max_w = doc.width
        max_h = 16 * cm

        def add_image(caption: str, path: str):
            if not os.path.exists(path):
                story.append(Paragraph(_format_inline_code(f"[пропущено] {caption}: файл не найден `{path}`"), base))
                story.append(Spacer(1, 6))
                return

            try:
                ir = ImageReader(path)
                iw, ih = ir.getSize()
                scale = min(max_w / float(iw), max_h / float(ih), 1.0)
                w = iw * scale
                h = ih * scale

                story.append(Paragraph(_format_inline_code(caption), h3))
                story.append(RLImage(path, width=w, height=h))
                story.append(Paragraph(_format_inline_code(f"`{os.path.basename(path)}`"), caption_style))
                story.append(Spacer(1, 10))
            except Exception as e:
                story.append(Paragraph(_format_inline_code(f"[ошибка] {caption}: {e}"), base))
                story.append(Spacer(1, 6))

        for cap, pth in images:
            add_image(cap, pth)

    doc.build(story)
    return pdf_path

def _safe_read_history_csv(run_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _plot_best_run_curves(best_run_dir: str, report_dir: str) -> List[Tuple[str, str]]:
    imgs: List[Tuple[str, str]] = []
    dfh = _safe_read_history_csv(best_run_dir)
    if dfh is None or dfh.empty:
        return imgs

    # Loss curve
    if "loss" in dfh.columns:
        fig = plt.figure(figsize=(7.5, 4.5))
        plt.plot(dfh["loss"].values, label="train_loss")
        if "val_loss" in dfh.columns:
            plt.plot(dfh["val_loss"].values, label="val_loss")
        plt.title("Loss по эпохам (лучший запуск)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        out = os.path.join(report_dir, "best_run_loss.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        imgs.append(("Кривые loss (лучший запуск)", out))

    # Accuracy curve
    if "accuracy" in dfh.columns:
        fig = plt.figure(figsize=(7.5, 4.5))
        plt.plot(dfh["accuracy"].values, label="train_accuracy")
        if "val_accuracy" in dfh.columns:
            plt.plot(dfh["val_accuracy"].values, label="val_accuracy")
        plt.title("Accuracy по эпохам (лучший запуск)")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        out = os.path.join(report_dir, "best_run_accuracy.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        imgs.append(("Кривые accuracy (лучший запуск)", out))

    return imgs


def _plot_lr_compare(df_runs: pd.DataFrame, report_dir: str) -> List[Tuple[str, str]]:
    imgs: List[Tuple[str, str]] = []
    compare = df_runs[df_runs["run_name"].astype(str).str.startswith("compare_")]
    if compare.empty:
        return imgs

    fig = plt.figure(figsize=(8.0, 4.8))
    plotted = 0
    for _, row in compare.iterrows():
        run_dir = str(row["run_dir"])
        name = str(row["run_name"])
        dfh = _safe_read_history_csv(run_dir)
        if dfh is None or "val_accuracy" not in dfh.columns:
            continue
        plt.plot(dfh["val_accuracy"].values, label=name)
        plotted += 1

    if plotted > 0:
        plt.title("Сравнение стратегий: val_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("val_accuracy")
        plt.legend()
        out = os.path.join(report_dir, "compare_lr_val_accuracy.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        imgs.append(("Сравнение LR-стратегий по val_accuracy", out))

    plt.close(fig)
    return imgs


def _find_latest_batch_profile_csv(root_dir: str) -> Optional[str]:
    candidates = glob(os.path.join(root_dir, "profile_batch_*", "batch_profile.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _plot_batch_profile(root_dir: str, report_dir: str) -> List[Tuple[str, str]]:
    imgs: List[Tuple[str, str]] = []
    path = _find_latest_batch_profile_csv(root_dir)
    if not path:
        return imgs

    try:
        df = pd.read_csv(path)
        if df.empty or "batch_size" not in df.columns:
            return imgs

        # epoch time vs batch size
        if "avg_epoch_time_sec" in df.columns:
            fig = plt.figure(figsize=(7.5, 4.5))
            d2 = df.sort_values("batch_size")
            plt.plot(d2["batch_size"].values, d2["avg_epoch_time_sec"].values, marker="o")
            plt.title("Время эпохи vs batch_size")
            plt.xlabel("batch_size")
            plt.ylabel("avg_epoch_time_sec")
            out = os.path.join(report_dir, "profile_epoch_time_vs_batch.png")
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)
            imgs.append(("Профилирование: время эпохи vs batch_size", out))

        # RSS peak vs batch size
        if "rss_peak_mb" in df.columns:
            fig = plt.figure(figsize=(7.5, 4.5))
            d2 = df.sort_values("batch_size")
            plt.plot(d2["batch_size"].values, d2["rss_peak_mb"].values, marker="o")
            plt.title("Память (RSS peak) vs batch_size")
            plt.xlabel("batch_size")
            plt.ylabel("rss_peak_mb")
            out = os.path.join(report_dir, "profile_rss_vs_batch.png")
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)
            imgs.append(("Профилирование: RSS peak vs batch_size", out))

        return imgs
    except Exception:
        return imgs


def _collect_images_for_pdf(root_dir: str, report_dir: str, df_runs: pd.DataFrame) -> List[Tuple[str, str]]:
    images: List[Tuple[str, str]] = []

    if df_runs.empty:
        return images

    # лучший запуск по test_accuracy
    best = df_runs.sort_values("test_accuracy", ascending=False).iloc[0]
    best_run_dir = str(best["run_dir"])

    # 1) Архитектура модели (если есть)
    arch_src = os.path.join(best_run_dir, "artifacts", "model_arch.png")
    if os.path.exists(arch_src):
        arch_dst = os.path.join(report_dir, "best_model_arch.png")
        try:
            shutil.copyfile(arch_src, arch_dst)
            images.append(("Архитектура модели (лучший запуск)", arch_dst))
        except Exception:
            pass

    # 2) Кривые обучения лучшего запуска
    images += _plot_best_run_curves(best_run_dir, report_dir)

    # 3) Сравнение стратегий LR (если были compare_ прогоны)
    images += _plot_lr_compare(df_runs, report_dir)

    # 4) Профилирование batch size (если запускалось)
    images += _plot_batch_profile(root_dir, report_dir)

    return images


def write_report(root_dir="runs") -> str:
    report_dir = os.path.join(root_dir, "_report")
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    ensure_dir(report_dir)

    df = summarize_runs(root_dir=root_dir)
    df_path = os.path.join(report_dir, "runs_summary.csv")
    df.to_csv(df_path, index=False)

    recs = generate_recommendations(df)

    # Теоретическое обоснование (кратко, по делу)
    theory = [
        "- Мониторинг loss/accuracy показывает динамику качества, но не объясняет причины проблем.",
        "- Гистограммы весов и градиентов позволяют диагностировать exploding/vanishing gradients и деградацию обучения.",
        "- Гистограммы активаций по слоям выявляют насыщение (например, tanh уходит в ±1) и “мертвые” слои.",
        "- Отслеживание дельты весов помогает понять, происходит ли реальное обучение или модель “застыла”.",
        "- ReduceLROnPlateau и LR-schedule ускоряют сходимость и повышают стабильность, early stopping предотвращает переобучение.",
        "- Confusion matrix и примеры предсказаний дают интерпретируемый срез ошибок модели."
    ]

    # Постановка эксперимента
    setup = [
        "- Датасет: MNIST, 28×28 → padding до 32×32, канал 1.",
        "- Архитектура: LeNet-5 (Conv→AvgPool→Conv→AvgPool→FC→FC→Softmax).",
        "- Логирование: TensorBoard (scalars + histograms) + custom callbacks (gradients, activations, deltas, confusion matrix, examples).",
        "- Сломанные сценарии: высокий LR, плохая инициализация, отсутствие нормализации.",
        "- Стратегии LR: fixed, step decay, cyclical (triangular).",
        "- Профилирование: время эпохи и память при разных batch_size; CPU↔GPU через отдельные процессы."
    ]

    # Сравнительный анализ
    analysis_lines = []
    if not df.empty:
        top = df.sort_values("test_accuracy", ascending=False).head(10)
        analysis_lines.append("Топ запусков по test_accuracy:")
        analysis_lines.append(top[["run_name", "lr_strategy", "test_accuracy", "best_val_accuracy", "avg_epoch_time_sec", "rss_peak_mb"]].to_string(index=False))

    report_md = os.path.join(report_dir, "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Отчет по экспериментам: LeNet-5 + TensorBoard (вариант 9)\n\n")

        f.write("## Теоретическое обоснование\n")
        for line in theory:
            f.write(f"{line}\n")
        f.write("\n")

        f.write("## Постановка эксперимента\n")
        for line in setup:
            f.write(f"{line}\n")
        f.write("\n")

        f.write("## Сводные результаты\n")
        f.write(f"- Таблица: `{os.path.basename(df_path)}`\n\n")

        f.write("## Сравнительный анализ\n")
        if analysis_lines:
            f.write("```text\n")
            for line in analysis_lines:
                f.write(line + "\n")
            f.write("```\n\n")
        else:
            f.write("Недостаточно данных для анализа (таблица пустая).\n\n")

        f.write("## Рекомендации по гиперпараметрам\n")
        for r in recs:
            f.write(f"- {r}\n")
        f.write("\n")

        f.write("## Как быстро развернуть аналогичный эксперимент\n")
        f.write("- Используйте `experiment_template.json` из нужного run_dir как стартовую конфигурацию.\n")
        f.write("- Меняйте: `lr.strategy`, `lr.lr/base_lr/max_lr`, `train.batch_size/epochs`, включайте/выключайте callbacks.\n\n")

        f.write("## Где смотреть визуализации\n")
        f.write("- Запустите TensorBoard: `tensorboard --logdir runs`\n")
        f.write("- Секции: Scalars (loss/accuracy/perf), Histograms (weights/grads/activations), Images (confusion matrix / examples), Graph.\n")

    # --- PDF generation ---
    pdf_path = os.path.join(report_dir, "report.pdf")
    try:
        images = _collect_images_for_pdf(root_dir=root_dir, report_dir=report_dir, df_runs=df)
        render_pdf_from_report_md(
            report_md_path=report_md,
            pdf_path=pdf_path,
            summary_csv_path=df_path,
            images=images
        )
    except Exception:
        err_path = os.path.join(report_dir, "pdf_error.txt")
        with open(err_path, "w", encoding="utf-8") as ef:
            ef.write(traceback.format_exc())

    return report_md



# 11) Главный сценарий


def default_full_suite(root_dir="runs") -> Dict[str, Any]:
    results = {"baseline": None, "broken": [], "compare_lr": [], "profile": None, "cpu_gpu": None, "report": None}

    # 1) baseline (расширенная телеметрия)
    cfg_baseline = ExperimentConfig(
        data=DataConfig(normalize=True),
        model=ModelConfig(initializer="glorot_uniform"),
        lr=LRConfig(strategy="fixed", lr=1e-3),
        train=TrainConfig(run_name="baseline_full_telemetry", epochs=8, batch_size=128, optimizer="adam")
    )
    results["baseline"] = run_experiment(cfg_baseline, root_dir=root_dir)

    # 2) broken suite
    results["broken"] = broken_suite(root_dir=root_dir)

    # 3) compare LR strategies
    results["compare_lr"] = compare_lr_strategies(root_dir=root_dir)

    # 4) profile batch sizes
    results["profile"] = profile_batch_sizes(root_dir=root_dir, batch_sizes=(32, 64, 128, 256), epochs=2)

    # 5) cpu vs gpu benchmark (опционально)
    try:
        results["cpu_gpu"] = cpu_gpu_benchmark(root_dir=root_dir)
    except Exception as e:
        # не считаем это фатальной ошибкой (например, если окружение ограничено)
        results["cpu_gpu"] = {"error": str(e), "gpus_visible": available_gpus()}

    # 6) report
    results["report"] = write_report(root_dir=root_dir)
    save_json(os.path.join(root_dir, "_report", "full_suite_results.json"), results)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="full",
                   choices=["full", "baseline", "broken", "compare_lr", "profile", "cpu_gpu_benchmark", "single_benchmark", "report_only"])
    p.add_argument("--root", type=str, default="runs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    return p.parse_args()


def main():
    args = parse_args()
    root_dir = args.root
    ensure_dir(root_dir)

    if args.mode == "single_benchmark":
        # internal use
        single_benchmark(args.device, root_dir=root_dir)
        return

    if args.mode == "baseline":
        cfg = ExperimentConfig(
            data=DataConfig(normalize=True),
            model=ModelConfig(initializer="glorot_uniform"),
            lr=LRConfig(strategy="fixed", lr=1e-3),
            train=TrainConfig(run_name="baseline_only", epochs=8, batch_size=128, optimizer="adam")
        )
        run_experiment(cfg, root_dir=root_dir)
        print(write_report(root_dir=root_dir))
        return

    if args.mode == "broken":
        broken_suite(root_dir=root_dir)
        print(write_report(root_dir=root_dir))
        return

    if args.mode == "compare_lr":
        compare_lr_strategies(root_dir=root_dir)
        print(write_report(root_dir=root_dir))
        return

    if args.mode == "profile":
        profile_batch_sizes(root_dir=root_dir, batch_sizes=(32, 64, 128, 256), epochs=2)
        print(write_report(root_dir=root_dir))
        return

    if args.mode == "cpu_gpu_benchmark":
        cpu_gpu_benchmark(root_dir=root_dir)
        print(write_report(root_dir=root_dir))
        return

    if args.mode == "report_only":
        print(write_report(root_dir=root_dir))
        return

    # default: full
    res = default_full_suite(root_dir=root_dir)
    print(f"Report: {res['report']}")


if __name__ == "__main__":
    main()
