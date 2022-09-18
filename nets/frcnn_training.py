import tensorflow as tf
import numpy as np


def rpn_cls_loss():
    """
    rpn只做二分类，是否有object
    二分类交叉熵损失
    :return:
    """

    def _rpn_cls_loss(y_true, y_pred):
        # ----------------------- #
        # y_true [batch_size,num_anchor,1]
        # y_pred [batch_size,num_anchor,1]
        # -1是要忽略的，0是背景，1是存在目标
        # ----------------------- #
        anchor_state = y_true
        # ----------------------- #
        # 获得无需忽略的所有样本
        # ----------------------- #
        indices_for_not_ignore = tf.where(tf.keras.backend.not_equal(anchor_state, -1))
        y_true_no_ignore = tf.gather_nd(y_true, indices_for_not_ignore)  # 1-D array
        y_pred_no_ignore = tf.gather_nd(y_pred, indices_for_not_ignore)  # 1-D array
        # ----------------------- #
        # 计算交叉熵
        # ----------------------- #
        y_true_no_ignore = tf.cast(y_true_no_ignore, dtype=tf.float32)
        y_pred_no_ignore = tf.cast(y_pred_no_ignore, dtype=tf.float32)
        cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true_no_ignore, y_pred_no_ignore)
        return cross_entropy_loss

    return _rpn_cls_loss


def rpn_smooth_l1(sigma=1.0):
    """
    rpn smooth l1 损失，只有正例和GT计算损失
    :param sigma:
    :return:
    """
    sigma_squared = sigma ** 2

    def _rpn_smooth_l1(y_true, y_pred):
        # ----------------------- #
        # y_true [batch_size, num_anchors, 4 + 1]
        # y_pred [batch_size, num_anchors, 4]
        # ----------------------- #
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        # ----------------------- #
        # -1是要忽略的，0是背景，1是存在目标
        # ----------------------- #
        anchor_state = y_true[:, :, -1]
        # ----------------------- #
        # 获取正样本
        # ----------------------- #
        indices = tf.where(tf.keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)  # 2-D array
        regression_target = tf.gather_nd(regression_target, indices)  # 2-D array
        # ----------------------- #
        # 计算smooth l1损失:
        # 0.5*x^2 if |x|<1
        # |x|-0.5 otherwise
        # ----------------------- #
        regression_diff = regression - regression_target
        x = tf.abs(regression_diff)
        loss_arr = tf.where(x < 1.0 / sigma_squared,
                            0.5 * sigma_squared * tf.pow(x, 2),
                            x - 0.5 / sigma_squared
                            )
        # 将loss全部加起来
        total_loss = tf.reduce_sum(loss_arr)
        # total_loss 除以样本数，计算平均loss
        num_indices = tf.maximum(1., tf.cast(tf.shape(indices)[0], tf.float32))
        avg_loss = total_loss / num_indices
        return avg_loss

    return _rpn_smooth_l1


def classifier_cls_loss():
    """
    最后分类的损失函数
    :return:
    """

    def _classifier_cls_loss(y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    return _classifier_cls_loss


def classifier_smooth_l1(num_classes, sigma=1.0):
    """
    最后框的回归损失函数
    :param num_classes:
    :param sigma:
    :return:
    """
    sigma_squared = sigma ** 2
    epsilon = 1e-4

    def _classifier_smooth_l1(y_true, y_pred):
        regression = y_pred
        # TODO 如果num_classes是20，为什么要取80之后的元素,80之前的元素是啥，答案：由于 roi_helper.calc_iou 生成的样本的前80是标签，后80才是坐标
        regression_target = y_true[:, :, 4 * num_classes:]

        regression_diff = regression_target - regression
        x = tf.abs(regression_diff)
        loss_arr = tf.where(x < 1 / sigma_squared,
                            0.5 / sigma_squared * tf.pow(x, 2),
                            x - 0.5 / sigma_squared
                            )
        # TODO 为什么要乘以 y_true[:, :, :4 * num_classes]，答案：因为只有正例的前80(20 * 4)对应位置是1，其余是0
        loss = tf.reduce_sum(loss_arr * y_true[:, :, :4 * num_classes]) * 4
        normalizer = tf.keras.backend.sum(epsilon + y_true[:, :, :4 * num_classes])
        loss = loss / normalizer
        return loss
    return _classifier_smooth_l1


class ProposalTargetCreator(object):
    """
    生成roi框
    """
    def __init__(self, num_classes, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5,
                 neg_iou_thresh_high=0.5, neg_iou_thresh_low=0, variance=[0.125, 0.125, 0.25, 0.25]):
        """
        :param num_classes: 类别数
        :param n_sample: 生成样本数
        :param pos_ratio: 正例的比例
        :param pos_iou_thresh: 和GT的IOU超过0.5就是正例
        :param neg_iou_thresh_high: IOU介于neg_iou_thresh_low和neg_iou_thresh_high之间的是负例
        :param neg_iou_thresh_low: IOU介于neg_iou_thresh_low和neg_iou_thresh_high之间的是负例
        :param variance:
        """
        self.num_classes = num_classes
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        # 正样本的数量
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low
        self.variance = variance

    def bbox_iou(self, bbox_a, bbox_b):
        """
        计算两个框的IOU，shape:[none, 4]
        :param bbox_a: 第一个框
        :param bbox_b: 第二个框
        :return: IOU的值
        """
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            print(bbox_a, bbox_b)
            raise ValueError
        # bbox_a[:, None, :2]中的None代表中间增加个维度
        top_left = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        bottom_right = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
        # 乘以np.all是为了把面积为负数的变为0，如果两个框不相交，则为负数，相交则为正数
        # 如果a的shape[5, 4],b的shape[3, 4],则intersection_area的shape[5,3]
        intersection_area = np.prod(bottom_right - top_left, axis=2) * np.all(bottom_right > top_left, axis=2)
        a_area = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)  # [5,]
        b_area = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)  # [3,]
        return intersection_area / (a_area[:, None] + b_area - intersection_area)  # [5, 3]

    def bbox2loc(self, src_bbox, dst_bbox):
        """
        计算回归损失中的t和t^*
        :param src_bbox:[none, 4],anchors boxes
        :param dst_bbox:[none, 4]
        :return:
        """
        # 计算anchor中心点坐标x,y以及宽和高w,h
        width = src_bbox[:, 2] - src_bbox[:, 0]
        height = src_bbox[:, 3] - src_bbox[:, 1]
        center_x = src_bbox[:, 0] + 0.5 * width
        center_y = src_bbox[:, 1] + 0.5 * height

        # 计算中心点坐标x,y以及宽和高w,h
        base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
        base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
        base_center_x = dst_bbox[:, 0] + 0.5 * base_width
        base_center_y = dst_bbox[:, 1] + 0.5 * base_height

        # 防止除0
        eps = np.finfo(height.dtype).eps
        width = np.maximum(width, eps)
        height = np.maximum(height, eps)

        t_x = (base_center_x - center_x) / width
        t_y = (base_center_y - center_y) / height
        t_w = np.log(base_width / width)
        t_h = np.log(base_height / height)

        t = np.stack([t_x, t_y, t_w, t_h], axis=1)
        return t

    def calc_iou(self, R, all_boxes):
        """
        划分正例和负例，用于后续训练
        :param R:建议框roi
        :param all_boxes:GT
        :return:
        """
        if len(all_boxes) == 0:
            max_iou = np.zeros(len(R))
            gt_assignment = np.zeros(len(R), dtype=np.int32)
            gt_roi_label = np.zeros(len(R))
        else:
            bboxes = all_boxes[:, :4]
            label = all_boxes[:, 4]
            R = np.concatenate([R, bboxes], axis=0)  # TODO 为什么这里要做拼接,答案：拼接起来一起确定正例和负例
            iou = self.bbox_iou(R, bboxes)  # [len(R) + len(bboxes), len(bboxes)]

            # 获得每个建议框roi最对应的真实框的iou
            max_iou = np.max(iou, axis=1)  # [len(R) + len(bboxes),] 又 [num_roi,]
            gt_assignment = np.argmax(iou, axis=1)  # [num_roi,]

            # 和哪个GT的IoU最大，标签就是哪个GT的标签，[num_roi,]
            gt_roi_label = label[gt_assignment]

        # ------------------------------------------------------------ #
        # 和GT的IoU大于pos_iou_thresh是正例
        # 将正例控制在n_sample/2之下,如果超过了则随机截取，如果不够则用负例填充
        # ------------------------------------------------------------ #
        pos_indices = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.n_sample // 2, pos_indices.size))
        if pos_indices.size > pos_roi_per_this_image:
            # replace：True可以重复先择,False不可以重复选择，元素不够则报错
            pos_indices = np.random.choice(pos_indices, size=pos_roi_per_this_image, replace=False)

        # ------------------------------------------------------------ #
        # 和GT的IoU大于neg_iou_thresh_low,小于neg_iou_thresh_high的作为负例
        # 正例数量和负例的数量相加等于n_sample
        # ------------------------------------------------------------ #
        neg_indices = np.where((max_iou >= self.neg_iou_thresh_low) & (max_iou < self.neg_iou_thresh_high))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        if neg_roi_per_this_image > neg_indices.size:
            neg_indices = np.random.choice(neg_indices, size=neg_roi_per_this_image, replace=True)
        else:
            neg_indices = np.random.choice(neg_indices, size=neg_roi_per_this_image, replace=False)

        # 正例和负例的索引
        keep_indices = np.append(pos_indices, neg_indices)
        # 保留下来的正负例框,[n_samples, 4]
        sample_roi = R[keep_indices]
        # 保留下来的框对应最大IoU的GT box,[n_samples, 4]
        gt_boxes = bboxes[gt_assignment[keep_indices]]

        if len(all_boxes) != 0:
            # 计算t_star
            gt_roi_loc = self.bbox2loc(sample_roi, gt_boxes)
            gt_roi_loc = gt_roi_loc / np.array(self.variance)  # TODO 这里为什么要进行缩放,答案：前面计算t都会进行缩放
        else:
            gt_roi_loc = np.zeros_like(sample_roi)

        # [128,]
        gt_roi_label = gt_roi_label[keep_indices]
        # 负例的标签置为20，正例的标签是0~19共20类
        gt_roi_label[pos_roi_per_this_image:] = self.num_classes - 1
        # ------------------------------------------------------------ #
        #   X   [n_sample, 4]
        #   Y1  [n_sample, num_classes] one_hot编码
        # ------------------------------------------------------------ #
        X = np.zeros_like(sample_roi)
        # roipooling中的tf.image.crop_and_resize需要这样的格式：[y_min,x_min,y_max,x_max]
        X[:, [0, 1, 2, 3]] = sample_roi[:, [1, 0, 3, 2]]
        Y1 = np.eye(self.num_classes)[np.array(gt_roi_label, np.int32)]

        # (n_sample=128, num_classes-1=20, 4)
        y_class_regr_label = np.zeros((self.n_sample, self.num_classes - 1, 4))
        y_class_regr_coords = np.zeros((self.n_sample, self.num_classes - 1, 4))
        y_class_regr_label[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image],
                           np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = 1
        y_class_regr_coords[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image],
                            np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = gt_roi_loc[:pos_roi_per_this_image]

        y_class_regr_label = np.reshape(y_class_regr_label, (self.n_sample, -1))  # (128,80)
        y_class_regr_coords = np.reshape(y_class_regr_coords, (self.n_sample, -1)) # (128,80)

        Y2 = np.concatenate([y_class_regr_label, y_class_regr_coords], axis=1)  # (128,160)
        return X, Y1, Y2






