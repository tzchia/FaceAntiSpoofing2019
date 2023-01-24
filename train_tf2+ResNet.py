import collections
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import app, flags
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow_addons.optimizers import LAMB, AdamW, LazyAdam

from src import data
from src.config import set_gpu_config
from src.data import AntiSpoofingDataset, Sensor
from src.resnet import *

FLAGS = flags.FLAGS
flags.DEFINE_string("task", None, "")
flags.DEFINE_string("root", None, "")
flags.DEFINE_integer("epoch", 2000, "")
flags.DEFINE_integer("bs", 256, "batch size")
flags.DEFINE_integer("gpu", 0, "specify gpu id")
flags.DEFINE_string("resume", None, "use checkpoint")
flags.DEFINE_enum('net', None, ['ResNet5', 'ResNet12'], 'net tag')
flags.DEFINE_string("optimizer", 'Adam', "")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("drop_f", 0.0, "Drop Fake Rate")
flags.DEFINE_boolean("summary", False, "show model summary")
flags.DEFINE_boolean("tensorboard", False, "use tensorboard callback")

flags.mark_flag_as_required('task')
flags.mark_flag_as_required('root')
flags.mark_flag_as_required('net')


def main(argv):
    set_gpu_config(gpus=str(FLAGS.gpu))
    validation_rate = 0

    ds_train = AntiSpoofingDataset(
        root=FLAGS.root,
        net='resnet12',
        shuffle=1,
        augment=1,
        drop_fake_rate=FLAGS.drop_f,
        batch_size=FLAGS.bs,
    ).dataset
    train(ds_train)


def train(ds):
    print("start train function")

    OUT_DIR = f'outputs/{datetime.today().strftime("%Y%m%d")}_{FLAGS.task}'
    log_dir = f"{OUT_DIR}/logs"
    shutil.copytree("src", log_dir)
    shutil.copy("train.py", log_dir)

    callbacks = []

    fold_dir = f"{OUT_DIR}/weights"
    os.makedirs(fold_dir)
    filepath = os.path.join(fold_dir, "weight.{epoch:05d}_{accuracy:.5f}.h5")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath)
    callbacks.append(ckpt_cb)

    # write command line to cmd.txt
    with open(f'{OUT_DIR}/CMD.txt', 'w') as f:
        print(' '.join(sys.argv), file=f)

    if FLAGS.tensorboard:
        log_dir = f"{OUT_DIR}/TensorBoard"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard_callback)

    if FLAGS.resume:
        print(f"Resume: {FLAGS.resume}")
        model = load_model(FLAGS.resume, compile=False)
    else:

        if FLAGS.net == 'ResNet12':
            model = ResNet12(input_shape=(32, 32, 3))
        
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
    if FLAGS.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(FLAGS.lr, momentum=0.9)
    elif FLAGS.optimizer == 'SGDPie':
        SCHEDULE_BOUNDARIES = [
            500,
            1000,
            1500,
        ]
        lr_schedule = PiecewiseConstantDecay(
            boundaries=SCHEDULE_BOUNDARIES,
            values=[
                FLAGS.lr,
                FLAGS.lr * 0.1,
                FLAGS.lr * 0.01,
                FLAGS.lr * 0.001,
            ],
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, 
                                            momentum=0.9)
    elif FLAGS.optimizer == 'LAMB':
        optimizer = LAMB(FLAGS.lr)
    elif FLAGS.optimizer == 'LazyAdam':
        optimizer = LazyAdam(FLAGS.lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if FLAGS.summary:
        model.summary()

    model.fit(ds, epochs=FLAGS.epoch, callbacks=callbacks)
    print("end train function")


if __name__ == "__main__":
    app.run(main)
