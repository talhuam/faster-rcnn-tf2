import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten


# -------------------- #
# 最后一个block没有做池化，输出是输入的1/16
# -------------------- #
def VGG16(inputs):
    # block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

    # block2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

    # block3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

    # block4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

    # block5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')

    return x


def VGG_Dense_layer(x):
    """
    roi_pooling之后
    :param x:
    :return:
    """
    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = TimeDistributed(Dense(4096, activation='relu'), name='fc1')(x)
    x = TimeDistributed(Dense(4096, activation='relu'), name='fc2')(x)
    return x
