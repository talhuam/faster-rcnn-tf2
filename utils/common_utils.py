import cv2
from PIL import Image
import numpy as np


def get_classes(path):
    with open(path, 'r', encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [line.strip() for line in class_names]  # 去除首尾空格
    return class_names


def cvt2RGB(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


'''def cvt2RGB(image):
    """
    opencv
    灰度图转RGB
    :param image:
    :return:
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image'''

'''def resize_image(image, size=(600, 600)):
    """
    opencv
    resize尺寸
    :param image:
    :param size:
    :return:
    """
    return cv2.resize(image, size)'''

def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(height * f)
        resized_width = int(width * f)
    else:
        f = float(img_min_side) / height
        resized_height = int(height * f)
        resized_width = int(width * f)
    return resized_height, resized_width
