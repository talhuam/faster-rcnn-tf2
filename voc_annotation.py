import xml.etree.ElementTree as ET
from utils.common_utils import get_classes
import os
import random

VOCdevkit_path = 'D:/datasets/VOCdevkit'
# -------------------------- #
# 0:分割xml数据集生成对应数据集文件 且 生成训练数据和验证数据文件
# 1:只划分数据集
# 2:只基于分割的数据集生成训练数据和验证数据文件
# -------------------------- #
annotation_mode = 0
# -------------------------- #
# trainval_percent指定训练集和验证集 与 测试集的比例，默认是9:1
# train_percent指定训练集和验证集的比例，默认是9:1
# -------------------------- #
trainval_percent = 0.9
train_percent = 0.9
# -------------------------- #
# 获取分类
# -------------------------- #
class_names = get_classes('model_data/voc_classes.txt')
# -------------------------- #
# 需要生成的数据，可选trainval/train/test/val
# -------------------------- #
sets = ['train', 'val']

if __name__ == '__main__':
    random.seed(666)
    if annotation_mode == 0 or annotation_mode == 1:
        print('split dataset begin')
        xml_path = os.path.join(VOCdevkit_path, 'Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'ImageSets/Main')
        annotation_files = os.listdir(xml_path)

        num_annotation_files = len(annotation_files)
        idxes = range(num_annotation_files)

        trainval_num = int(num_annotation_files * trainval_percent)
        train_num = int(trainval_num * train_percent)
        trainval = random.sample(idxes, trainval_num)
        train = random.sample(trainval, train_num)

        trainval_file = open(os.path.join(saveBasePath, 'trainval.txt'), 'w', encoding='utf-8')
        train_file = open(os.path.join(saveBasePath, 'train.txt'), 'w', encoding='utf-8')
        test_file = open(os.path.join(saveBasePath, 'test.txt'), 'w', encoding='utf-8')
        val_file = open(os.path.join(saveBasePath,'val.txt'), 'w', encoding='utf-8')

        for i in idxes:
            name = annotation_files[i][:-4] + '\n'
            if i in trainval:
                trainval_file.write(name)
                if i in train:
                    train_file.write(name)
                else:
                    val_file.write(name)
            else:
                test_file.write(name)

        trainval_file.close()
        train_file.close()
        test_file.close()
        val_file.close()
        print('split dataset end')

    if annotation_mode == 0 or annotation_mode == 2:
        print('Generating train.txt and val.txt for train')
        for s in sets:
            # 读取分割的数据集来生成训练的数据
            with open(os.path.join(VOCdevkit_path, 'ImageSets/Main/%s.txt' % s)) as f:
                image_ids = f.readlines()
            image_ids = [i.strip() for i in image_ids]

            data_file = open('%s.txt' % s, 'w', encoding='utf-8')
            for image_id in image_ids:
                data_file.write(os.path.abspath(VOCdevkit_path).replace('\\', '/') + '/JPEGImages/' + image_id + '.jpg')
                # 解析xml
                annotation_file = open(os.path.join(VOCdevkit_path, 'Annotations/%s.xml' % image_id))
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    difficult = 0
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in class_names or int(difficult) == 1:
                        continue
                    cls_id = class_names.index(cls)
                    obj_box = obj.find('bndbox')
                    box_coordinate = [obj_box.find('xmin').text,
                                      obj_box.find('ymin').text,
                                      obj_box.find('xmax').text,
                                      obj_box.find('ymax').text]
                    data_file.write(' ' + ','.join(box_coordinate)+',' + str(cls_id))
                data_file.write('\n')
            data_file.close()


