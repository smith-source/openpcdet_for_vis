import cv2, math
import numpy as np


def AddHaze1(img):
    img_f = img
    (row, col, chs) = img.shape

    A = 0.9  # 亮度
    beta = 0.1  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f


def AddHaze2(img):
    A = np.random.uniform(0.6, 0.95)
    t = np.random.uniform(0.3, 0.95)
    img_h = img * t + A * (1 - t)

    return img_h


def AddHaze(img):
    img = img/255
    l = np.random.uniform(0, 1)
    if l > 0.7:
        img_T = AddHaze1(img)
    else:
        img_T = AddHaze2(img)
    return img_T*255


def demo(img):

    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.32  # 亮度
    beta = 0.04  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    return img_f*255



def main():
    path = '/home/smith/my_projects/data/kitti/training/image_2/006704.png'
    img = cv2.imread(path)
    # cv2.namedWindow("fog")
    fog_img = demo(img)
    # cv2.imshow("fog", fog_img)
    cv2.imwrite('006704_test_file_img.png', fog_img)

if __name__ == "__main__":
    main()