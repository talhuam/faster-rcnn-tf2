import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.common_utils import get_classes
from utils.anchor_utils import get_anchors
from nets.frcnn import get_model
from utils.callbacks import LossHistory
from utils.bbox_utils import BBoxUtility
from nets.frcnn_training import ProposalTargetCreator
from nets.frcnn_training import rpn_cls_loss, rpn_smooth_l1, classifier_cls_loss, classifier_smooth_l1
from utils.dataloader import FRCNNDatasets
from utils.fit_utils import fit_one_epoch
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if __name__ == '__main__':

    # ------------------------------------------- #
    # 输入图片的大小
    # ------------------------------------------- #
    INPUT_SHAPE = (600, 600)
    # ------------------------------------------- #
    # 类别文件
    # ------------------------------------------- #
    CLASSES_PATH = 'model_data/voc_classes.txt'
    # ------------------------------------------- #
    # 权值文件
    # ------------------------------------------- #
    MODEL_PATH = 'model_data/voc_weights_vgg.h5'
    # model_path = ''
    # ------------------------------------------- #
    # anchor_box尺寸
    # ------------------------------------------- #
    ANCHORS_SIZE = [128, 256, 512]
    ANCHORS_RATIO = [[1, 1], [1, 2], [2, 1]]
    # ------------------------------------------- #
    # 冻结阶段,0~50冻结训练
    # 此时模型的主干被冻结了，特征提取的网络不发生改变
    # ------------------------------------------- #
    INIT_EPOCH = 0
    FREEZE_EPOCH = 50
    FREEZE_BATCH_SIZE = 4
    FREEZE_LR = 1e-4
    # ------------------------------------------- #
    # 解冻阶段,50~100解冻训练
    # 此时模型的主干提取网络解冻，特征提取网络发生变化
    # ------------------------------------------- #
    UNFREEZE_EPOCH = 100
    UNFREEZE_BATCH_SIZE = 2
    UNFREEZE_LR = 1e-5
    # ------------------------------------------- #
    # 是否进行冻结训练，默认先冻结主干训练后解冻训练
    # ------------------------------------------- #
    FREEZE_TRAIN = True
    # ------------------------------------------- #
    # 图片路径和标签
    # ------------------------------------------- #
    TRAIN_ANNOTATION_PATH = 'train.txt'
    VAL_ANNOTATION_PATH = 'val.txt'

    # 获取classes和anchor
    class_names = get_classes(CLASSES_PATH)
    num_classes = len(class_names) + 1  # 加上背景(background)
    anchors = get_anchors(INPUT_SHAPE, ANCHORS_SIZE, ANCHORS_RATIO, stride=16)

    # 构建模型
    model_rpn, model_all = get_model(num_classes, len(ANCHORS_SIZE) * len(ANCHORS_RATIO))

    # 加载权重
    if MODEL_PATH != '':
        model_rpn.load_weights(MODEL_PATH, by_name=True)
        model_all.load_weights(MODEL_PATH, by_name=True)

    # callback 回调
    callback = tf.summary.create_file_writer('logs')
    loss_history = LossHistory('logs/')

    # 含有非极大值抑制
    bbox_util = BBoxUtility(num_classes)
    # 含有rpn损失和最终损失，划分正例和负例
    roi_helper = ProposalTargetCreator(num_classes)

    # 读取数据集对应的txt文件
    with open(TRAIN_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(VAL_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # -------------------------------------------------------------------------- #
    # *主干冻结训练阶段*
    # -------------------------------------------------------------------------- #
    # vgg主干17层，如果包含Input层是18层
    freeze_layers = 18
    if FREEZE_TRAIN:
        for i in range(freeze_layers):
            if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
                model_all.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers'.format(freeze_layers, len(model_all.layers)))
    # ------------------------------------------ #
    # 主干特征提取网络可以通用，冻结训练可以加快训练速度，
    # 也可以防止训练初期权值被破坏
    # ------------------------------------------ #
    if True:
        batch_size = FREEZE_BATCH_SIZE
        lr = FREEZE_LR
        start_epoch = INIT_EPOCH
        end_epoch = FREEZE_EPOCH

        # 一个epoch多个batch_size
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集')

        # 模型编译
        model_rpn.compile(loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1()
        }, optimizer=tf.keras.optimizers.Adam(lr))
        model_all.compile(loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes): classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes): classifier_smooth_l1(num_classes - 1)
        }, optimizer=tf.keras.optimizers.Adam(lr))

        # ----------------------------------------------------- #
        # 1.解析文本数据，读取图片[resize,数据增强(可选,训练数据增强,验证数据不增强)]生成image_data，解析坐标和类别标签生成boxes
        # 2.正负例划分
        #   a.与GT的IoU大于0.7的作为正例，标签值为1
        #   b.与GT的IoU在0.3~0.7之间的忽略，标签值为-1
        #   c.小于0.3的是负例，标签值为0
        # 3.计算正例的t_star，即回归损失中的t_star
        # ----------------------------------------------------- #
        gen = FRCNNDatasets(train_lines, INPUT_SHAPE, anchors, batch_size, num_classes, train=False).generate()
        gen_val = FRCNNDatasets(val_lines, INPUT_SHAPE, anchors, batch_size, num_classes, train=False).generate()
        print('Train on {} samples, val on {} samples, with batch size {}'.format(num_train, num_val, batch_size))

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          end_epoch, anchors, bbox_util, roi_helper)
            lr = lr * 0.96
            tf.keras.backend.set_value(model_rpn.optimizer.lr, lr)
            tf.keras.backend.set_value(model_all.optimizer.lr, lr)

    # -------------------------------------------------------------------------- #
    # *主干解冻训练阶段*
    # -------------------------------------------------------------------------- #
    if FREEZE_TRAIN:
        for i in range(freeze_layers):
            if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
                model_all.layers[i].trainable = True

    if True:
        batch_size = UNFREEZE_BATCH_SIZE
        lr = UNFREEZE_LR
        start_epoch = FREEZE_EPOCH
        end_epoch = UNFREEZE_EPOCH

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集')

        # 模型编译
        model_rpn.compile(loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1()
        }, optimizer=tf.keras.optimizers.Adam(lr))
        model_all.compile(loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes): classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes): classifier_smooth_l1(num_classes - 1)
        }, optimizer=tf.keras.optimizers.Adam(lr))

        gen = FRCNNDatasets(train_lines, INPUT_SHAPE, anchors, batch_size, num_classes, train=False).generate()
        gen_val = FRCNNDatasets(val_lines, INPUT_SHAPE, anchors, batch_size, num_classes, train=False).generate()
        print('Train on {} samples, val on {} samples, with batch size {}'.format(num_train, num_val, batch_size))

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          end_epoch, anchors, bbox_util, roi_helper)
            lr = lr * 0.96
            tf.keras.backend.set_value(model_rpn.optimizer.lr, lr)
            tf.keras.backend.set_value(model_all.optimizer.lr, lr)
