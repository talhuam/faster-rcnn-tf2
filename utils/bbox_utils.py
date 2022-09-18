import numpy as np
import tensorflow as tf


class BBoxUtility(object):

    def __init__(self, num_classes, rpn_pre_boxes=12000, rpn_nms=0.7, nms_iou=0.3, min_k=300):
        # 分类数
        self.num_classes = num_classes
        # 非极大值抑制前框的数量
        self.rpn_pre_boxes = rpn_pre_boxes
        # 非极大值抑制的IoU,超过该值就丢弃，应用于RPN之后
        self.rpn_nms = rpn_nms
        # 非极大值抑制的IoU,超过该值就丢弃，应用于网络输出之后
        self.nms_iou = nms_iou
        # 极大值抑制后框的数量
        self.min_k = min_k

    def decode_boxes(self, mbox_loc, anchors, variances):
        """
        从t_x,t_y,t_w,t_h 解析成 xmin,ymin,xmax,ymax
        是 nets.frcnn_training.ProposalTargetCreator.bbox2loc 的逆过程
        :param mbox_loc:
        :param anchors:
        :param variances:
        :return:
        """
        # 获得先验框的宽和高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心
        anchor_center_x = anchors[:, 0] + 0.5 * anchor_width
        anchor_center_y = anchors[:, 1] + 0.5 * anchor_height

        # 真实框距离先验框中心xy轴的偏移情况
        detections_center_x = mbox_loc[:, 0] * anchor_width * variances[0] + anchor_center_x
        detections_center_y = mbox_loc[:, 1] * anchor_width * variances[1] + anchor_center_y
        # 真实框的宽和高的求取
        detections_width = np.exp(mbox_loc[:, 2] * variances[2]) * anchor_width
        detections_height = np.exp(mbox_loc[:, 3] * variances[3]) * anchor_height

        # 真实框的左上角和右下角
        detections_xmin = detections_center_x - detections_width * 0.5
        detections_ymin = detections_center_y - detections_height * 0.5
        detections_xmax = detections_center_x + detections_width * 0.5
        detections_ymax = detections_center_y + detections_height * 0.5

        detections = np.stack([
            detections_xmin,
            detections_ymin,
            detections_xmax,
            detections_ymax
        ], axis=1)
        # 防止超过0和1
        detections = np.minimum(np.maximum(detections, 0), 1)
        return detections

    def detection_out_rpn(self, predictions, anchors, variances=[0.25, 0.25, 0.25, 0.25]):
        """
        rpn网络预测之后调用，只取置信度前12000的框进行非极大抑制
        :param predictions:
        :param anchors:
        :param variances:
        :return:
        """
        # 种类的置信度,[batch_size, 37*37*9, 1]
        mbox_conf = predictions[0]
        # 回归预测结果,[batch_size, 37*37*9, 4]
        mbox_loc = predictions[1]

        result = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以循环只进行过一次
        for i in range(len(mbox_loc)):
            # --------------------------- #
            # 利用回归结果对先验框解码
            # --------------------------- #
            detections = self.decode_boxes(mbox_loc[i], anchors, variances)
            # --------------------------- #
            # 去除先验框中包含物体的概率,在依据置信度进行从大到小排序
            # --------------------------- #
            c_conf = mbox_conf[i, :, 0]
            c_conf_argsort = np.argsort(c_conf)[::-1][:self.rpn_pre_boxes]
            # --------------------------- #
            # 原始的预测框太多，选取前12000的高分框进行处理
            # --------------------------- #
            confs_to_process = c_conf[c_conf_argsort]
            boxes_to_process = detections[c_conf_argsort, :]
            # --------------------------- #
            # 进行非极大抑制
            # --------------------------- #
            idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self.min_k, self.rpn_nms).numpy()
            # --------------------------- #
            # 取出非极大抑制效果最好的框
            # --------------------------- #
            good_boxes = boxes_to_process[idx]
            result.append(good_boxes)

        return result

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # -----------------------------------------------------------------#
        # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # 因为input_shape和image_shape的格式是(h,w)
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mines = box_yx - (box_hw / 2)
        box_maxes = box_yx + (box_hw / 2)
        boxes = np.concatenate([box_mines[..., 0:1], box_mines[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        # 归一化的逆过程
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def detection_out_classifier(self, predictions, rpn_results, image_shape, input_shape, confidence=0.5, variance=[0.125, 0.125, 0.25, 0.25]):
        """
        预测结果之后调用，都预测框进行解码并进行非极大值抑制
        :param predictions: classifier 预测结果
        :param rpn_results: rpn网络预测结果
        :param image_shape: 图片原始尺寸
        :param input_shape: 图片调整后的尺寸，最短边是600
        :param confidence: 只有置信度大于0.5的框才保留下来
        :param variance:
        :return:
        """
        # 置信度
        proposal_conf = predictions[0]
        # 回归预测结果
        proposal_loc = predictions[1]

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以循环只进行过一次
        for i in range(len(proposal_conf)):
            results.append([])
            detections = []
            # ------------------------------------------- #
            # 计算建议框中心和宽高
            # ------------------------------------------- #
            rpn_results[i, :, 2] = rpn_results[i, :, 2] - rpn_results[i, :, 0]
            rpn_results[i, :, 3] = rpn_results[i, :, 3] - rpn_results[i, :, 1]
            rpn_results[i, :, 0] = rpn_results[i, :, 0] + rpn_results[i, :, 2] / 2
            rpn_results[i, :, 1] = rpn_results[i, :, 1] + rpn_results[i, :, 3] / 2

            for j in range(proposal_conf[i].shape[0]):
                # 获得第j个建议框的置信度
                score = np.max(proposal_conf[i][j, :-1])
                # 获得label标签
                label = np.argmax(proposal_conf[i][j, :-1])
                if score < confidence:
                    # 小于confidence的框直接忽略
                    continue

                x, y, w, h = rpn_results[i, j, :]
                t_x, t_y, t_w, t_h = proposal_loc[i, j, 4 * label: 4 * label + 4]

                # t_x,t_y,t_w,t_h 转化为 x_min,y_min.x_max,y_max
                center_x = t_x * variance[0] * w + x
                center_y = t_y * variance[1] * h + y
                width = np.exp(t_w * variance[2]) * w
                height = np.exp(t_h * variance[3]) * h

                x_min = center_x - width / 2.
                y_min = center_y - height / 2.
                x_max = center_x + width / 2
                y_max = center_y + height / 2.

                detections.append([x_min, y_min, x_max, y_max, score, label])
            detections = np.array(detections)

            # 进行非极大抑制
            if len(detections) > 0:
                for c in range(self.num_classes):
                    c_confs_mask = detections[:, -1] == c
                    if len(detections[c_confs_mask] > 0):
                        boxes_to_process = detections[:, :4][c_confs_mask]
                        confs_to_precess = detections[:, 4][c_confs_mask]
                        # ---------------------------------------- #
                        # 相同类别的进行NMS非极大值抑制
                        # ---------------------------------------- #
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_precess, self.min_k,
                                                           iou_threshold=self.nms_iou).numpy()
                        # ---------------------------------------- #
                        # 取出非极大值抑制之后的框
                        # extend 2-D -> 多个1-D
                        # ---------------------------------------- #
                        results[-1].extend(detections[c_confs_mask][idx])

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = \
                    (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results