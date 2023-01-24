import random
from enum import Enum
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow._api.v2 import data


class Sizes(Enum):
    default = [300,300,32]

class AntiSpoofingDataset():
    def __init__(self,
                 root,
                 sizes=Sizes.default,
                 net='resnet12',
                 shuffle=False,
                 augment=False,
                 include_path=False,
                 validation_rate=0,
                 batch_size=32):
        self.root = root
        self.s1 = sizes.value[0]
        self.s2 = sizes.value[1]
        self.s3 = sizes.value[2]
        self.net = net
        self.shuffle = shuffle
        self.augment = augment
        self.include_path = include_path
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.dataset = self._load_dataset_V2()

    def _load_dataset_V2(self):
        if self.net == 'resnet12':
            pattern = '*.png'
        else:
            raise RuntimeError('Error self.net attribute, [resnet5, resnet7]')

        #raw_files = [self.root]
        raw_files = sorted(list(glob(f'{self.root}/**/{pattern}', recursive=True)))
        if self.shuffle:
            random.shuffle(raw_files)
        files = []
        labels = []
        for f in raw_files:
            if '/real/' in f:
                files.append(f)
                labels.append(1)
            elif '/fake/' in f:
                files.append(f)
                labels.append(0)
            else:
                raise RuntimeError('Error format of label')
        files = np.array(files)
        labels = np.array(labels)

        print(f'{self.root}: ',
              f'Real -> {np.count_nonzero(labels == 1):,}, ',
              f'Fake -> {np.count_nonzero(labels == 0):,}')

        if self.validation_rate > 0:
            k = int(len(labels) * self.validation_rate)
            indicies = list(range(labels.shape[0]))
            if self.shuffle:
                random.shuffle(indicies)
            train_files = files[indicies[k:]]
            train_labels = labels[indicies[k:]]
            val_files = files[indicies[:k]]
            val_labels = labels[indicies[:k]]
            print(f'├── Training Set: ',
                f'Real -> {np.count_nonzero(train_labels == 1):,}, ',
                f'Fake -> {np.count_nonzero(train_labels == 0):,}')
            print(f'└── Validation Set: ',
                f'Real -> {np.count_nonzero(val_labels == 1):,}, ',
                f'Fake -> {np.count_nonzero(val_labels == 0):,}')
            dataset_train = self._convert_tf_data(train_files, train_labels)
            self.augment = False
            self.include_path = True
            dataset_val = self._convert_tf_data(val_files, val_labels)
            return dataset_train, dataset_val
        elif self.validation_rate < 0 or type(self.validation_rate) != int:
            raise RuntimeError('Invalid value')
        else:
            if len(files) > 0:
                dataset = self._convert_tf_data(files, labels)
                return dataset
            else:
                pass

    def _convert_tf_data(self, files, labels):
        labels = tf.keras.utils.to_categorical(
            labels, num_classes=2, dtype='int')

        dataset = tf.data.Dataset.from_tensor_slices((files, labels))

        dataset = dataset.map(self._preprocessing,
                              tf.data.experimental.AUTOTUNE)
        #print(list(dataset.as_numpy_iterator()))
        
        dataset = dataset.cache()

        if self.augment and not self.include_path:
            dataset = dataset.map(self._data_augmentation,
                                  tf.data.experimental.AUTOTUNE)

        if self.net == 'resnet12' and self.s1 != self.s3:
            if self.include_path:
                dataset = dataset.map(lambda f, x, y: (f, tf.image.resize_with_crop_or_pad(x, self.s3, self.s3), y),
                                      tf.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.map(lambda x, y: (tf.image.resize_with_crop_or_pad(x, self.s3, self.s3), y),
                                      tf.data.experimental.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=65536)

        dataset = dataset.batch(self.batch_size)

        return dataset

    def _preprocessing(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = img / 255

        if self.include_path:
            return file_path, img, label
        else:
            return img, label

    def _gaussian_blur(self, img, lower, upper):
        # sigma = tf.random.uniform(shape=[],
        #                           minval=min_sigma,
        #                           maxval=max_sigma,
        #                           dtype=tf.float32)
        # sigma = random_ops.random_uniform([], lower, upper)
        # https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
        sigma = random.uniform(lower, upper)
        k = 2 * tf.math.ceil(3 * sigma) + 1
        return tfa.image.gaussian_filter2d(img, filter_shape=(k, k), sigma=sigma)

    def _random_adjust_contrast(self, img, lower, upper):
        d1, d2 = tf.unstack(img, axis=-1)
        d1 = tf.expand_dims(d1, axis=-1)
        d2 = tf.expand_dims(d2, axis=-1)
        contrast_factor = tf.random.uniform(shape=[],
                                            minval=lower,
                                            maxval=upper)
        d1 = tf.image.adjust_contrast(d1, contrast_factor)
        d2 = tf.image.adjust_contrast(d2, contrast_factor)

        return tf.concat([d1, d2], axis=-1)

    def _cut_out(self, img, mask_size=(8, 8), counts=1):
        for _ in range(counts):
            img = tfa.image.random_cutout(tf.expand_dims(img, 0), mask_size)
            img = tf.squeeze(img)

        return img

    def _data_augmentation(self, img, label):
        img = tf.image.random_flip_left_right(img)
        img = self._cut_out(img, mask_size=(8, 8), counts=3)

        if self.s2 != self.s3:
            img = tf.image.random_crop(img, size=img_shape)

        # img = tf.image.random_brightness(img, max_delta=0.2)
        # img = tf.image.random_contrast(img, lower=0.8, upper=1.5)

        # noise = tf.random.normal(shape=tf.shape(img)) / 40
        # img = tf.add(img, noise)

        img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)

        return img, label
