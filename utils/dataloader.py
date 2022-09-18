import math
import cv2
import numpy as np
from utils.common_utils import cvt2RGB, resize_image
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input


class FRCNNDatasets():
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, num_sample=256,
                 ignore_threshold=0.3, overlap_threshold=0.7):
        self.length = len(annotation_lines)
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        # 是否数据增强，True数据增强，False不进行数据增强，只resize大小
        self.train = train
        self.num_sample = num_sample
        self.ignore_threshold = ignore_threshold
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        # 向上取整
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def generate(self):
        i = 0
        while True:
            image_datas = []
            classifications = []
            regressions = []
            targets = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.annotation_lines)

                # 训练时对数据进行随机增强，验证时不对数据进行增强
                image, box = self.process_data(self.annotation_lines[i], self.input_shape, random=self.train)
                if len(box) != 0:
                    # TODO 为什么这里又要除以相应的宽和高，答案：utils.anchor_utils.get_anchors 做了归一化，为了统一
                    boxes = np.array(box[:, :4], np.float32)
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
                    box = np.concatenate([boxes, box[:, -1:]], axis=-1)

                # [num_anchors, 4+1],4代表正例(>overlap_threshold)计算的t_star,1取值有(-1,1,0)
                assignment = self.assign_boxes(box)
                classification = assignment[:, 4]
                regression = assignment[:, :]

                # --------------------------------------- #
                # 对正样本和负样本进行筛选，训练样本之和为256
                # --------------------------------------- #
                pos_idx = np.where(classification > 0)[0]
                num_pos = len(pos_idx)
                if num_pos > self.num_sample // 2:
                    num_pos = self.num_sample // 2
                    disable_index = np.random.choice(pos_idx, size=(len(pos_idx) - self.num_sample // 2), replace=False)
                    classification[disable_index] = -1
                    regression[disable_index, -1] = -1

                neg_idx = np.where(classification == 0)[0]
                num_neg = self.num_sample - num_pos
                if len(neg_idx) > num_neg:
                    disable_index = np.random.choice(neg_idx, size=(len(neg_idx) - num_neg), replace=False)
                    classification[disable_index] = -1
                    regression[disable_index, -1] = -1

                i = (i + 1) % self.length
                image_datas.append(preprocess_input(image))
                classifications.append(np.expand_dims(classification, -1))
                regressions.append(regression)
                targets.append(box)

            # ---------------------------------- #
            # image_data [batch_size,600,600,3]
            # classifications [batch_size,num_anchors,1]
            # regressions [batch_size,num_anchors,5]
            # ---------------------------------- #
            yield np.array(image_datas), [np.array(classifications, dtype=np.float32),
                                          np.array(regressions, dtype=np.float32)], targets

    def process_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """
        对行数据进行处理，生成处理后的图片和GT框
        :param annotation_line: train.txt或者val.txt中的一行数据
        :param input_shape: 目标尺寸，例(600, 600)
        :param jitter:宽高扭曲比率
        :param hue:色调
        :param sat:饱和度
        :param val:亮度
        :param random:是否图象增强，以上四个为数据增强参数
        :return:boxes shape is [num_gt,5], image_data shape is [input_shape[0],input_shape[1],3]
        """
        fields = annotation_line.split()
        # --------------------------- #
        # 读取图像并转化为RGB格式
        # --------------------------- #
        image = Image.open(fields[0])
        image = cvt2RGB(image)
        iw, ih = image.size
        w, h = input_shape
        # --------------------------- #
        # 获得预测框
        # --------------------------- #
        boxes = np.array([f.split(',') for f in fields[1:]], dtype=np.int_)

        if not random:
            # ---------------------------------- #
            # 图片尺寸计算，图片resize宽和高的比例不变，
            # 不足的部分用灰条填充
            # ---------------------------------- #
            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = resize_image(image, (nw, nh))
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            # dx和dy为左上角的坐标
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            # ---------------------------------- #
            # 对真实框进行调整
            # ---------------------------------- #
            if len(boxes) > 0:
                np.random.shuffle(boxes)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (nw / iw) + dx
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (nh / ih) + dy
                # x_min, y_min小于0的等于0;x_max,y_max超出了宽和高的等于宽和高
                boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
                boxes[:, 2][boxes[:, 2] > w] = w
                boxes[:, 3][boxes[:, 3] > h] = h
                boxes_width = boxes[:, 2] - boxes[:, 0]
                boxes_height = boxes[:, 3] - boxes[:, 1]
                # 只保留宽和高同时大于1的框
                boxes = boxes[np.logical_and(boxes_width > 1, boxes_height > 1)]

            return image_data, boxes

        else:
            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # ------------------------------------------#
            #   将图像多余的部分加上灰条
            # ------------------------------------------#
            dx = int(self.rand(0, w - nw))
            dy = int(self.rand(0, h - nh))
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image

            # ------------------------------------------#
            #   翻转图像
            # ------------------------------------------#
            flip = self.rand() < .5
            if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # ------------------------------------------#
            #   色域扭曲
            # ------------------------------------------#
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(boxes) > 0:
                np.random.shuffle(boxes)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
                if flip: boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
                boxes[:, 2][boxes[:, 2] > w] = w
                boxes[:, 3][boxes[:, 3] > h] = h
                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]
            return image_data, boxes

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def assign_boxes(self, boxes):
        """
        划分数据集，
        介于ignore_threshold(0.3)和overlap_threshold(0.7)之间的需要忽略，标签为-1
        大于overlap_threshold(0.7)是含有object，标签是1
        其他的标签是0，没有含有object
        :param boxes:ground truth boxes
        :return:
        """
        # ------------------------------ #
        # 主干网络输出为37*37的特征图,每个点有9个框，num_anchors=37*37*9
        # ------------------------------ #
        assignment = np.zeros(shape=(self.num_anchors, 4 + 1))
        assignment[:, 4] = 0
        if len(boxes) == 0:
            return assignment

        # [[array(), array()],......]
        result = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([result[i, 0] for i in range(len(result))])
        ignored_boxes = np.array([result[i, 1] for i in range(len(result))])
        # ------------------------------ #
        # ignored_boxes reshape之后的shape为
        # [num_true_box, num_anchors, 1],其中1是IoU
        # 其中有值的iou是介于ignore_threshold与overlap_threshold之间
        # ------------------------------ #
        ignored_boxes = ignored_boxes.reshape(-1, self.num_anchors, 1)
        # 取出每个anchor与gt最大的iou,
        # 只要与一个gt的iou介于ignore_threshold与overlap_threshold之间，标签都会置为-1，后续计算会忽略掉
        ignore_iou = np.max(ignored_boxes[:, :, 0], axis=0)
        ignore_iou_mask = ignore_iou > 0
        assignment[:, 4][ignore_iou_mask] = -1

        # ------------------------------ #
        # encode_boxes reshape之后的shape为
        # [num_true_box, num_anchors, 4+1]
        # 其中有值的iou大于overlap_threshold
        # ------------------------------ #
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        best_iou = np.max(encoded_boxes[:, :, 4], axis=0)
        best_iou_idx = np.argmax(encoded_boxes[:, :, 4], axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # 有多少先验框满足需求
        num_assign = len(best_iou_idx)

        # 将编码后的真实框(iou>overlap_threshold)取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(num_assign), :4]
        # -1表示忽略，1表示有object，0表示没有object
        assignment[:, 4][best_iou_mask] = 1
        return assignment

    def iou(self, box):
        """
        计算当前真实框(GT)和每个先眼框(anchors)的IoU
        :param box: ground truth，shape：[4,]
        :return: shape is [num_anchors,]
        """
        # ----------------------------- #
        # 计算真实框和先验框的重合度，即交集的面积
        # ----------------------------- #
        left_top = np.maximum(self.anchors[:, :2], box[:2])
        right_bottom = np.minimum(self.anchors[:, 2:4], box[2:])
        intersection_wh = right_bottom - left_top
        intersection_wh = np.maximum(intersection_wh, 0)
        intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]
        # ----------------------------- #
        # 计算真实框的面积
        # ----------------------------- #
        gt_wh = box[2:] - box[:2]
        gt_area = gt_wh[0] * gt_wh[1]
        # ----------------------------- #
        # 计算每个先眼框的面积
        # ----------------------------- #
        anchors_wh = self.anchors[:, 2:] - self.anchors[:, :2]
        anchors_area = anchors_wh[:, 0] * anchors_wh[:, 1]
        return intersection_area / (gt_area + anchors_area - intersection_area)

    def encode_ignore_box(self, box, return_iou=True, variance=[0.25, 0.25, 0.25, 0.25]):
        """
        1.计算介于ignore_threshold和overlap_threshold框的IoU
        2.计算大于overlap_threshold框的t_star和IoU,如果没有大于overlap_threshold的框，则用IoU最大的框作为正例
        :param box: 真实框
        :param return_iou:
        :param variance:
        :return:
        """
        # 计算当前真实框和先验框的IoU
        iou = self.iou(box)

        ignored_box = np.zeros(shape=(self.num_anchors, 1))
        # 找到iou处于ignore_threshold和overlap_threshold之间iou
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]
        # 4 + True = 5, 4 + False = 4
        encoded_box = np.zeros(shape=(self.num_anchors, 4 + return_iou))

        # 找到IoU大于overlap_threshold
        assign_mask = iou > self.overlap_threshold
        # 如果没有一个IoU大于overlap_threshold，则选取iou最大的作为正样本
        if not assign_mask.any():
            assign_mask[np.argmax(iou)] = True

        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 找到iou超过overlap_threshold的anchor
        assigned_anchors = self.anchors[assign_mask]

        # ----------------------------------------------------------------- #
        # 将 真实框 和 重合度高的先验框 转化为FRCNN预测结果的格式：[center_x, center_y, width, height]
        # ----------------------------------------------------------------- #
        # 真实框转化
        box_center = 0.5 * (box[2:] + box[:2])
        box_wh = box[2:] - box[:2]
        # 先验框转化
        assigned_anchors_center = 0.5 * (assigned_anchors[:, 2:4] + assigned_anchors[:, :2])
        assigned_anchors_wh = assigned_anchors[:, 2:4] - assigned_anchors[:, :2]
        # ----------------------------------------------------------------- #
        # 计算t_star
        # ----------------------------------------------------------------- #
        encoded_box[:, :2][assign_mask] = (box_center - assigned_anchors_center) / assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variance)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh/assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variance)[2:]
        # np.ravel()：转化为一维数组
        return encoded_box.ravel(), ignored_box.ravel()
