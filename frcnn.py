import time
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.common_utils import get_classes, get_new_img_size, cvt2RGB, resize_image
from utils.bbox_utils import BBoxUtility
from utils.anchor_utils import get_anchors
import colorsys
import os
from nets.frcnn import get_predict_model
from PIL import ImageDraw, ImageFont


class FRCNN(object):
    _defaults = {
        # ---------------------------------- #
        # 训练自己的数据需要更改 model_path 和 classes_path
        # ---------------------------------- #
        'model_path': 'model_data/voc_weights_vgg.h5',
        'classes_path': 'model_data/voc_classes.txt',
        # 未实现两种网络，仅支持vgg
        'backbone': 'vgg',
        # 只有置信度大于confidence的框才保留下来
        'confidence': 0.5,
        # 模型输出还需要经过一次nms非极大值抑制，nms_iou为定义的阈值
        'nms_iou': 0.3,
        # 先验框的尺寸
        'anchors_size': [128, 256, 512]
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return f'Unrecognized attribute name <{n}>'

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 获取类别
        self.class_names = get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.num_classes += 1

        # 用于解码
        self.bbox_util = BBoxUtility(self.num_classes, nms_iou=self.nms_iou, min_k=150)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        # 在unix或者windows上，将参数开头的~或~user替换成当前用户的家目录并返回
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'keras model or weights mush be a .h5 file'
        # 获取模型拓扑
        self.model_rpn, self.model_classifier = get_predict_model(self.num_classes)
        # 加载权值
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True)
        print(f'{model_path} model, anchors, and classes loaded.')

    def detect_image(self, image, crop=False):
        # 获取图片的高和宽,(h, w)
        image_shape = np.shape(image)[:2]
        # 计算输入到网络中进行运算的图片的高和宽，保证最短边是600,(h, w)
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        # 将图片转化成RGB
        image = cvt2RGB(image)
        # resize图片
        image_data = resize_image(image, (input_shape[1], input_shape[0]))
        # 添加一个维度，batch_size的维度
        image_data = preprocess_input(np.array(image_data, np.float32))[None, :]
        # rpn 预测 以及 base_layer
        rpn_predict = self.model_rpn(image_data)
        rpn_predict = [x.numpy() for x in rpn_predict]
        # 获得先验框
        anchors = get_anchors(input_shape)
        # 由于rpn网络预测出来的是t_x,t_y,t_w,t_h,需要转为x_min,y_min,x_max,y_max，并进行非极大值抑制
        rpn_results = self.bbox_util.detection_out_rpn(rpn_predict, anchors)
        rpn_results = np.array(rpn_results)
        # 利用建议框获得最终的预测结果
        base_layer = rpn_predict[2]
        classifier_pred = self.model_classifier([base_layer, rpn_results[:, :, [1, 0, 3, 2]]])
        classifier_pred = [x.numpy() for x in classifier_pred]
        # 预测出来的是t_x, t_y, t_w, t_h, 需要转为x_min, y_min, x_max, y_max，并进行非极大值抑制
        # 返回box的坐标格式y_min,x_min,y_max,x_max
        results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                          self.confidence)
        if len(results[0]) == 0:
            return image

        # 最终的结果
        top_label = np.array(results[0][:, 5], dtype=np.int32)
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        # 设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // input_shape[0], 1)

        # ---------------------------------------------------------#
        # 是否对目标进行裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in enumerate(top_label):
                # 越界处理
                y_min, x_min, y_max, x_max = top_boxes[i]
                y_min = max(0, np.floor(y_min).astype(np.int32))
                x_min = max(0, np.floor(x_min).astype(np.int32))
                y_max = max(image.size[1], np.floor(y_max).astype(np.int32))
                x_max = max(image.size[0], np.floor(x_max).astype(np.int32))

                dir_save_path = 'img_crop'
                if not os.path.exists(dir_save_path):
                    os.mkdirs(dir_save_path)
                crop_image = image.crop([x_min, y_min, x_max, y_max])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        # 图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            class_name = self.class_names[top_label[i]]
            box = top_boxes[i]
            score = top_conf[i]

            y_min, x_min, y_max, x_max = box
            top = max(0, np.floor(y_min).astype(np.int32))
            left = max(0, np.floor(x_min).astype(np.int32))
            bottom = min(image.size[1], np.floor(y_max).astype(np.int32))
            right = min(image.size[0], np.floor(x_max).astype(np.int32))

            label = '{} {:.2f}'.format(class_name, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(f'label:{label},top[y_min]:{top},left[x_min]:{left},bottom[y_max]:{bottom},right[x_max]:{right}')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        # 获取图片的高和宽,(h, w)
        image_shape = np.shape(image)[:2]
        # 计算输入到网络中进行运算的图片的高和宽，保证最短边是600,(h, w)
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        # 将图片转化成RGB
        image = cvt2RGB(image)
        # resize图片
        image_data = resize_image(image, (input_shape[1], input_shape[0]))
        # 添加一个维度，batch_size的维度
        image_data = preprocess_input(np.array(image_data, np.float32))[None, :]

        t1 = time.time()
        for _ in range(test_interval):
            # ---------------------------------------------------------#
            # 获得rpn网络预测结果和base_layer
            # ---------------------------------------------------------#
            rpn_pred = self.model_rpn(image_data)
            rpn_pred = [x.numpy() for x in rpn_pred]
            # ---------------------------------------------------------#
            # 生成先验框并解码
            # ---------------------------------------------------------#
            anchors = get_anchors(input_shape)
            rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
            rpn_results = np.array(rpn_results)
            temp_ROIs = rpn_results[:, :, [1, 0, 3, 2]]

            # -------------------------------------------------------------#
            #   利用建议框获得classifier网络预测结果
            # -------------------------------------------------------------#
            classifier_pred = self.model_classifier([rpn_pred[2], temp_ROIs])
            classifier_pred = [x.numpy() for x in classifier_pred]
            # -------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            # -------------------------------------------------------------#
            results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                              self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

