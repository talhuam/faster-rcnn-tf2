# -----------------------------#
# 定义网络的结构，model_rpn & model_all
# -----------------------------#
from tensorflow.keras.layers import Input, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

from nets.vgg16 import VGG16, VGG_Dense_layer
from nets.roipooling import RoiPooling
from nets.rpn import RPN


def get_model(num_classes=21, num_anchors=9):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    # ------------------------- #
    # backbone：主干特征提取
    # 输入为600,600,3，输出为37,37,512
    # ------------------------- #
    base_layer = VGG16(inputs)
    # ------------------------- #
    # rpn网络
    # ------------------------- #
    rpn = RPN(base_layer, num_anchors)
    model_rpn = Model(inputs, rpn)

    # batch_size, num_rois, 7, 7, 512
    roi_pooling_out = RoiPooling(pool_size=7)([base_layer, roi_input])
    # batch_size, num_rois, 4096
    out = VGG_Dense_layer(roi_pooling_out)

    cls = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02)),
                          name='dense_class_{}'.format(num_classes))(out)
    reg = TimeDistributed(
        Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02)),
        name='dense_regress_{}'.format(num_classes))(out)

    # ------------------------- #
    # + 号会融合公共部分也就是base_layer共享特征层
    # 将两部分结合
    # ------------------------- #
    model_all = Model([inputs, roi_input], rpn + [cls, reg])

    return model_rpn, model_all


def get_predict_model(num_classes=21, num_anchors=9):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 512))

    base_layer = VGG16(inputs)
    rpn = RPN(base_layer, num_anchors)
    model_rpn = Model(inputs, rpn + [base_layer])

    roi_pooling_out = RoiPooling(pool_size=7)([feature_map_input, roi_input])

    out = VGG_Dense_layer(roi_pooling_out)

    cls = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02)),
                          name='dense_class_{}'.format(num_classes))(out)
    reg = TimeDistributed(
        Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02)),
        name='dense_regress_{}'.format(num_classes))(out)

    model_classifier_only = Model([feature_map_input, roi_input], [cls, reg])

    return model_rpn, model_classifier_only


if __name__ == '__main__':
    import tensorflow as tf
    import os

    # train
    model_rpn, model_all = get_model()
    tf.keras.utils.plot_model(model_rpn, show_shapes=True, show_layer_names=True, show_dtype=True,
                              to_file=f'{os.path.expanduser("~/Desktop/model_rpn.png")}')
    tf.keras.utils.plot_model(model_all, show_shapes=True, show_layer_names=True, show_dtype=True,
                              to_file=f'{os.path.expanduser("~/Desktop/model_all.png")}')

    # predict
    model_rpn, model_classifier_only = get_predict_model()
    tf.keras.utils.plot_model(model_rpn, show_shapes=True, show_layer_names=True, show_dtype=True,
                              to_file=f'{os.path.expanduser("~/Desktop/model_rpn_predict.png")}')
    tf.keras.utils.plot_model(model_classifier_only, show_shapes=True, show_layer_names=True, show_dtype=True,
                              to_file=f'{os.path.expanduser("~/Desktop/model_classifier_only.png")}')


