# coding=utf-8
'''将二进制格式的MNIST数据集转成.jpg图片格式并保存，图片标签包含在图片名中'''

import numpy as np
import cv2
import os
import struct

def save_mnist_to_jpg(mnist_image_file, mnist_label_file, save_dir):
    if 'train' in os.path.basename(mnist_image_file):
        prefix = 'train'
    else:
        prefix = 'test'

    labelIndex = 0
    imageIndex = 0
    i = 0
    lbdata = open(mnist_label_file, 'rb').read()
    magic, nums = struct.unpack_from(">II", lbdata, labelIndex)
    labelIndex += struct.calcsize('>II')

    imgdata = open(mnist_image_file, "rb").read()
    magic, nums, numRows, numColumns = struct.unpack_from('>IIII', imgdata, imageIndex)
    imageIndex += struct.calcsize('>IIII')

    for i in range(nums):
        label = struct.unpack_from('>B', lbdata, labelIndex)[0]
        labelIndex += struct.calcsize('>B')
        im = struct.unpack_from('>784B', imgdata, imageIndex)
        imageIndex += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        img = im.reshape(28, 28)
        save_name = os.path.join(save_dir, '{}.{}.jpg'.format(label,i))
        cv2.imwrite(save_name, img)


if __name__ == '__main__':
    train_images = r'..\source_data\mnist\train-images.idx3-ubyte'  # 训练集图像的文件名
    train_labels = r'..\source_data\mnist\train-labels.idx1-ubyte'  # 训练集label的文件名
    test_images = r'..\source_data\mnist\t10k-images.idx3-ubyte'  # 测试集图像的文件名
    test_labels = r'..\source_data\mnist\t10k-labels.idx1-ubyte'  # 测试集label的文件名

    save_train_dir = r'..\source_data\mnist\train'
    save_test_dir = r'..\source_data\mnist\test'

    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    save_mnist_to_jpg(test_images, test_labels, save_test_dir)
    save_mnist_to_jpg(train_images, train_labels, save_train_dir)