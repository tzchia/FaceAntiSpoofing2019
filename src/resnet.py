from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import AdaptiveAveragePooling2D


def ResidualBlock(x, filters, stride=1, no=''):
    identity = x

    x = layers.ZeroPadding2D(padding=1, name=f'Block{no}_pad1')(x)
    x = layers.Conv2D(filters,
                      kernel_size=3,
                      strides=stride,
                      padding='valid',
                      use_bias=False,
                      name=f'Block{no}_conv1')(x)
    x = layers.BatchNormalization(momentum=0.9,
                                  epsilon=1e-5,
                                  name=f'Block{no}_bn1')(x)
    x = layers.ReLU(name=f'Block{no}_relu1')(x)

    x = layers.ZeroPadding2D(padding=1, name=f'Block{no}_pad2')(x)
    x = layers.Conv2D(filters,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      use_bias=False,
                      name=f'Block{no}_conv2')(x)
    x = layers.BatchNormalization(momentum=0.9,
                                  epsilon=1e-5,
                                  gamma_initializer='zeros',
                                  name=f'Block{no}_bn2')(x)

    if stride > 1:
        identity = layers.Conv2D(filters,
                                 kernel_size=1,
                                 strides=stride,
                                 padding='valid',
                                 use_bias=False,
                                 name=f'Block{no}_conv1x1')(identity)
        identity = layers.BatchNormalization(momentum=0.9,
                                             epsilon=1e-5,
                                             name=f'Block{no}_bn3')(identity)

    x = layers.Add(name=f'Block{no}_add')([x, identity])
    x = layers.ReLU(name=f'Block{no}_relu2')(x)

    return x


def ResNet12(input_shape):
    inputs = keras.Input(shape=input_shape, name='input')

    # stem
    x = layers.ZeroPadding2D(padding=3)(inputs)
    x = layers.Conv2D(32,
                      kernel_size=7,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9,
                                  epsilon=1e-5,
                                  name='bn')(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.MaxPool2D(pool_size=(3, 3),
                         strides=2)(x)

    x = ResidualBlock(x, 32, no='1')
    x = ResidualBlock(x, 32, no='2')
    x = ResidualBlock(x, 64, stride=2, no='3')
    x = ResidualBlock(x, 64, no='4')
    x = ResidualBlock(x, 64, no='5')

    x = AdaptiveAveragePooling2D((1, 1))(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(2, name='dense')(x)

    model = keras.Model(inputs=inputs, outputs=x, name='ResNet10')

    return model
