import os
import tensorflow as tf


def set_gpu_config(log_level='2', gpu_growth=True, gpus=''):
    '''
    set gpu's config
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
    if gpu_growth:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if gpus != '':
        # gpus_list = tf.config.list_physical_devices('GPU')
        # if gpus_list:
        #     # Restrict TensorFlow to only use the first GPU
        #     try:
        #         tf.config.set_logical_device_configuration(gpus_list[int(gpus)],
        #                                                    [tf.config.LogicalDeviceConfiguration(memory_limit=3251)])
        #         tf.config.set_visible_devices(gpus_list[int(gpus)], 'GPU')
        #     except RuntimeError as e:
        #         # Visible devices must be set before GPUs have been initialized
        #         print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
