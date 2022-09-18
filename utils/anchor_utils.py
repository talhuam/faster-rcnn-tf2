import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_anchors(sizes=[128, 256, 512], ratios=[[1, 1], [1, 2], [2, 1]]):
    """
    生成基础的9个不同尺寸不同比例的框
    :param sizes:
    :param ratios:
    :return:
    """
    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = np.tile(sizes, [2, len(ratios)]).T

    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, [2, 1]).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, [2, 1]).T
    return anchors


def shift(feature_shape, anchors, stride=16):
    """
    对基础的先验框扩展获得所有的先验框
    :param feature_shape:
    :param anchors:
    :param stride:
    :return:
    """
    shift_x = (np.arange(0, feature_shape[1], dtype=np.float_) + 0.5) * stride
    shift_y = (np.arange(0, feature_shape[0], dtype=np.float_) + 0.5) * stride
    # 框中心点的位置
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 在一维的shift_x中的元素和shift_y中的元素一一对应形成坐标的x和y
    shift_x = np.reshape(shift_x, (-1))
    shift_y = np.reshape(shift_y, (-1))

    # 将shift_x和shift_y堆叠两次，分别来调整左上角和右下角的坐标
    shifts = np.stack([
        shift_x, shift_y, shift_x, shift_y
    ], axis=0)
    shifts = np.transpose(shifts)
    num_anchors = anchors.shape[0]  # 9

    k = shifts.shape[0]  # 37*37 = 1369
    # 对应维度广播，生成先验框在原图上的左上角和右下角的坐标，shape:[1369(7*7),9,4]
    shifted_anchors = np.reshape(anchors, (1, num_anchors, 4)) + tf.reshape(shifts, (k, 1, 4))
    # (1369 * 9,4)
    shifted_anchors = np.reshape(shifted_anchors, [k * num_anchors, 4])

    '''# 绘制先验框
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(shift_x, shift_y)
    plt.xlim(-300, 900)
    plt.ylim(-300, 900)
    shifted_widths = shifted_anchors[:, 2] - shifted_anchors[:, 0]
    shifted_heights = shifted_anchors[:, 3] - shifted_anchors[:, 1]
    init = 0
    for i in [init * 9 + i for i in range(9)]:
        # 左下角的x,y坐标，width,height
        rec = plt.Rectangle((shifted_anchors[i, 0], shifted_anchors[i, 1]), shifted_widths[i], shifted_heights[i], color='r', fill=False)
        ax.add_patch(rec)
    plt.show()'''

    return shifted_anchors


def get_anchors(input_shape, sizes=[128, 256, 512], ratios=[[1, 1], [1, 2], [2, 1]], stride=16):
    # ------------------------ #
    # vgg16作为主干特征提取网络，最后一次池化没有做，故而是原有的尺寸的1/16
    # 输入如果是600 * 600,则feature_shape就是37 * 37
    # ------------------------ #
    feature_shape = (int(input_shape[0] / 16), int(input_shape[1] / 16))
    anchors = generate_anchors(sizes, ratios)
    anchors = shift(feature_shape, anchors, stride=stride)
    # TODO 为什么这里要做归一化?，答案：因为 utils.dataloader.FRCNNDatasets.generate 在处理真实框的时候也做了归一化
    anchors = anchors.copy()
    anchors[:, 0::2] /= input_shape[1]
    anchors[:, 1::2] /= input_shape[0]
    anchors = np.clip(anchors, 0, 1)

    return anchors



