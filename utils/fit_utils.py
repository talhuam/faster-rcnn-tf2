import tensorflow as tf
import numpy as np
from tqdm import tqdm


def write_log(callback, names, valus, batch_no):
    with callback.as_default():
        for name, value in zip(names, valus):
            tf.summary.scalar(name, value, step=batch_no)
            callback.flush()


def fit_one_epoch(model_rpn, model_all, loss_history, callback, current_epoch, epoch_step, epoch_step_val, gen, gen_val,
                  end_epoch, anchors, bbox_utils, roi_helper):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_loss = 0
    with tqdm(total=epoch_step, desc=f'Epoch {current_epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            # -------------------------------------------------- #
            # X 是图片数据 [batch_size,600,600,3]
            # Y 两部分组成
            #   classification [batch_size,num_anchors,1] 取值0负例，1正例，-1忽略，正例负例相加为256
            #   regression [batch_size,num_anchors,4+1] 4代表t_x,t_y,t_w,t_h,1的值与上一致
            # gt_boxes [np.array(), np.array, ......],每个array的shape是[num_gt,4+1],4是归一化的左上角右下角饺坐标,1是具体的了类别
            # -------------------------------------------------- #
            X, Y, gt_boxes = batch[0], batch[1], batch[2]
            # -------------------------------------------------- #
            # predict_rpn包含两部分
            # x_cls [batch_size,37*37*9,1]
            # x_reg [batch_size,37*37*9,4]
            # -------------------------------------------------- #
            predict_rpn = model_rpn.predict_on_batch(X)
            # -------------------------------------------------- #
            # 1.先将t_x,t_y,t_w,t_h还原成x_min,y_min,x_max,y_max
            # 2.再从37*37*9的先验框中依据置信度从大到小排列选取前12000的先验框进行NMS非极大值抑制，NMS阈值0.7
            # result 包含一个批次多张图片的处理结果[np.array,np.array,......]
            # -------------------------------------------------- #
            results = bbox_utils.detection_out_rpn(predict_rpn, anchors)

            roi_inputs = []
            out_classes = []
            out_regrs = []
            for i in range(len(X)):
                R = results[i]
                # -------------------------------------------------- #
                # 1. 正负例样本的划分，正负例总数128，正负例各占一半，正例如果不足一般，用负例填充
                # 2. 与GT计算IoU，大于0.5的作为正例，介于0到0.5之间的作为负例，负例标签是20，正例标签为0~19
                # 3. X2 [128, 4] 4的格式是y_min,x_min,y_max,x_max,因为roipooling需要这样的格式
                # 4. Y1 [128, 21] one_hot编码
                # 5. Y2 [128, 160] 160中前80是0或1，后80是对应类别索引的4个值是t_x,t_y,t_w,t_h
                # -------------------------------------------------- #
                X2, Y1, Y2 = roi_helper.calc_iou(R, gt_boxes[i])
                roi_inputs.append(X2)
                out_classes.append(Y1)
                out_regrs.append(Y2)

            losses = model_all.train_on_batch([X, np.array(roi_inputs)],
                                              [Y[0], Y[1], np.array(out_classes), np.array(out_regrs)])
            write_log(callback,
                      ['total_loss', 'rpn_cls_loss', 'rpn_reg_loss', 'detection_cls_loss', 'detection_reg_loss'],
                      losses, iteration)

            rpn_cls_loss += losses[1]
            rpn_loc_loss += losses[2]
            roi_cls_loss += losses[3]
            roi_loc_loss += losses[4]
            total_loss = rpn_cls_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss

            pbar.set_postfix(**{
                'total': total_loss / (iteration + 1),
                'rpn_cls': rpn_cls_loss / (iteration + 1),
                'rpn_loc': rpn_loc_loss / (iteration + 1),
                'roi_cls': roi_cls_loss / (iteration + 1),
                'roi_loc': roi_loc_loss / (iteration + 1),
                'lr': tf.keras.backend.get_value(model_rpn.optimizer.lr)
            })
            pbar.update(1)

    print('start validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {current_epoch + 1}/{end_epoch}', postfix=dict,
              mininterval=0.3) as pbar:
        for interation, batch in enumerate(gen_val):
            if interation >= epoch_step_val:
                break
            X, Y, gt_boxes = batch[0], batch[1], batch[2]
            predict_rpn = model_rpn.predict_on_batch(X)

            results = bbox_utils.detection_out_rpn(predict_rpn, anchors)
            roi_inputs = []
            out_classes = []
            out_regrs = []

            for i in len(X):
                R = results[i]
                X2, Y1, Y2 = roi_helper.calc_iou(R, gt_boxes[i])
                roi_inputs.append(X2)
                out_classes.append(Y1)
                out_regrs.append(Y2)

            losses = model_all.train_on_batch([X, np.array(roi_inputs)],
                                              [Y[0], Y[1], np.array(out_classes), np.array(out_regrs)])
            val_loss += losses[0]
            pbar.set_postfix(**{
                'total': val_loss / (iteration + 1)
            })
            pbar.update(1)

    logs = {'loss': total_loss / epoch_step, 'val_loss': val_loss / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:' + str(current_epoch + 1) + '/' + str(end_epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    # 每个epoch都保存一次权值
    model_all.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5'
                           % (current_epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
