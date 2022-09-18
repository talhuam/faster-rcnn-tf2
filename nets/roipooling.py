# ----------------------------- #
# 定义roipooling层
# RoiPooling层将roi和feature map相结合
# ----------------------------- #

from tensorflow.keras.layers import Layer
import tensorflow as tf


class RoiPooling(Layer):

    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(RoiPooling, self).__init__(**kwargs)

    def get_config(self):
        """
        保存model需要实现该方法，否则报错
        :return:
        """
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size
        })
        return config

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        input_shape2 = input_shape[1]
        return None, input_shape2[1], self.pool_size, self.pool_size, self.nb_channels

    def call(self, inputs, **kwargs):
        assert (len(inputs) == 2)
        # -------------------------- #
        # 共享特征层
        # batch_size, 37, 37, 512
        # -------------------------- #
        feature_map = inputs[0]
        # -------------------------- #
        # 建议框
        # batch_size, num_rois, 4
        # -------------------------- #
        rois = inputs[1]
        # -------------------------- #
        # 获取roi的个数和batch_size
        # -------------------------- #
        batch_size = tf.shape(rois)[0]
        num_rois = tf.shape(rois)[1]
        # ---------------------------------#
        # 生成建议框序号信息,
        # 用于在进行crop_and_resize时
        # 帮助建议框找到对应的共享特征层
        # ---------------------------------#
        box_index = tf.expand_dims(tf.range(batch_size), 1)
        box_index = tf.tile(box_index, [1, num_rois])
        box_index = tf.reshape(box_index, [-1])
        # ---------------------------------#
        # boxes是二维的，[[ymin，xmin，ymax，xmax],......]
        # box_indices是一维的，如[0,0,0,1,1,1],帮助roi找到共享特征层,前三个box属于第0号feature_map,依次类推
        # ---------------------------------#
        roi_result = tf.image.crop_and_resize(feature_map, boxes=tf.reshape(rois, (-1, 4)), box_indices=box_index,
                                              crop_size=(self.pool_size, self.pool_size))

        # ---------------------------------#
        # 输出大小
        # batch_size,num_rois,7,7,512
        # ---------------------------------#
        final_output = tf.reshape(roi_result, (batch_size, num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output


'''
# crop_and_resize示例：
img = plt.imread('./VOCdevkit/JPEGImages/000001.jpg')
shape = img.shape
img = tf.reshape(img, (1, shape[0], shape[1], shape[2]))
a = tf.image.crop_and_resize(img, [[0.5, 0.6, 0.9, 0.8], [0.2, 0.6, 1.3, 0.9]], [0,0],(100,100))
print(a[0].shape)
plt.imshow(a[0] / 255)
plt.show()
'''
