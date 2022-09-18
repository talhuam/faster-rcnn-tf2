# -----------------------------#
# 定义RPN网络结构
# -----------------------------#

from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.initializers import RandomNormal


def RPN(base_layers, num_anchors):
    # --------------------------------- #
    # 先基于共享特征层做3*3的卷积
    # --------------------------------- #
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02),
               name='rpn_conv1')(base_layers)

    # --------------------------------- #
    # 再基于1*1的卷积调整通道数
    # x_cls [batch_size,37,37,9]
    # x_reg [batch_size,37,37,36]
    # --------------------------------- #
    x_cls = Conv2D(num_anchors, (1, 1), padding='same', activation='sigmoid',
                   kernel_initializer=RandomNormal(stddev=0.02), name='rpn_out_class')(x)
    x_reg = Conv2D(num_anchors * 4, (1, 1), padding='same', activation='linear',
                   kernel_initializer=RandomNormal(stddev=0.02), name='rpn_out_regress')(x)

    # --------------------------------- #
    # 再基于1*1的卷积调整通道数
    # x_cls [batch_size,37*37*9,1]
    # x_reg [batch_size,37*37*9,4]
    # --------------------------------- #
    x_cls = Reshape((-1, 1), name='classification')(x_cls)
    x_reg = Reshape((-1, 4), name='regression')(x_reg)

    return [x_cls, x_reg]
