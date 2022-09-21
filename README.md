## faster-rcnn tensorflow2实现
---
## 博客地址
https://blog.csdn.net/IT142546355/article/details/126897693

## 模型架构
![image](https://github.com/talhuam/faster-rcnn-tf2/blob/main/faster-rcnn-structure.png)

## 权重文件
链接：https://pan.baidu.com/s/19AEjI9vI3RBFeyjic4QzYA 
提取码：sqoz

## 所需环境
tensorflow-gpu==2.4.0

## 训练步骤
1.准备好数据集，数据集的结构需要和voc数据集的格式一致。即annotation xml文件需要放在VOCdevkit/Annotations下，图片需要放在VOCdevkit\JPEGImages下,如：D:/datasets/VOCdevkit  
  
2.修改voc_annotation.py中的VOCdevkit_path配置，指向VOCdevkit这一级目录；修改classes_path配置，指向分类文件；首次运行annotation_mode=0，即需要分割数据集并且生成训练时用到的train.txt和val.txt，annotation_mode=1只会划分数据集，annotation=2则基于划分的数据集生成train.txt和val.txt  
  
3.修改train.py中的CLASSES_PATH指向分类文件，MODEL_PATH指向权值文件，其他参数可看注释进行调整  
  
4.训练完之后权值文件保存在logs中，需要修改根目录下的frcnn.py中的model_path指向logs中的权值文件，classes_path指向分类文件，然后运行predict.py即可进行目标检测，有多种模式可选，
具体的可以参阅注释

## Reference
https://github.com/bubbliiiing/faster-rcnn-tf2
