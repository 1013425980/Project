"""
增加算子大小为3x3
流程：计算摩擦力场：考虑标准差和动态范围，计算归一化的摩擦力场
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# /-----------------------------------------------------------------------------------------------
# 图像展示
def cv_show(img, name='image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 绘制高斯分布图像
def gaussian_draw(size=512):
    center = size // 2
    sigma = 80  # 控制高斯分布的宽度

    # 创建网格
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    # 计算高斯分布
    A = 255
    gaussian_image = A * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    gaussian_image = gaussian_image.astype(np.uint8)  # 转换为8位灰度图

    # 向高斯图像添加噪声
    noise = np.random.normal(0, 15, gaussian_image.shape)
    noisy_image = np.clip(gaussian_image + noise, 0, 255).astype(np.uint8)
    return gaussian_image, noisy_image

# 反射边界填充
def reflection_padding(img, pad_size):
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    return padded_img

# 计算摩擦系数friction
def friction_calculate(img_gray, method='DYNAMIC_RANGE'):
    i_start, i_end, i_step, j_start, j_end, j_step = [0, img_gray.shape[0], 1, 0, img_gray.shape[1], 1]
    friction = np.zeros_like(img_gray, dtype=np.float32)            # 初始化速度场
    img_gray = reflection_padding(img_gray, 1)                      # 反射填充图像

    for i in range(i_start+1, i_end, i_step):
        for j in range(j_start+1, j_end, j_step):
            # 获取窗口
            window = img_gray[i-1:i+2, j-1:j+2]

            # 计算窗口归一化的标准差和动态范围
            std = np.std(window) / 100.0
            dynamic_range = (np.max(window) - np.min(window)) / 255.0

            # 计算摩擦系数
            friction[i-1, j-1] = 0.5 * std + 0.5 * dynamic_range
            friction[i-1, j-1] = friction[i-1, j-1] * 255.0

    return friction

# 计算速度场
def vociety_calculate(img_gray, friction, initial_velocity, acceleration, method='left'):
    # 预处理
    th = 0.3
    i_start, i_end, i_step, j_start, j_end, j_step = [0, img_gray.shape[0], 1, 0, img_gray.shape[1], 1]
    vociety_field = np.zeros_like(img_gray).astype(np.float32)# 初始化速度场
    vociety_field[:, 0] = initial_velocity
    img_gray = reflection_padding(img_gray, 1)  # 反射填充图像
    if method == 'left':
        pass
    elif method == 'right':
        img_gray = np.fliplr(img_gray)  # 左右翻转图像
    elif method == 'top':
        img_gray = img_gray.T      # 转置图像
    elif method == 'bottom':
        img_gray = np.fliplr(img_gray.T)  # 左右翻转并转置图像

    # 计算速度场
    for i in range(i_start+1, i_end, i_step):
        for j in range(j_start+1, j_end, j_step):
            # window = img_gray[i-1:i+2, j-1:j+2]
            # friction = np.max(window) - np.min(window)
            if vociety_field[i, j-1] < initial_velocity:
                vociety_field[i, j] = vociety_field[i, j-1] + (acceleration - friction[i, j]) * dt
            else:
                vociety_field[i, j] = vociety_field[i, j-1] - friction[i, j] * dt
            vociety_field[i, j] = np.clip(vociety_field[i, j], 0, initial_velocity)
        
            # 速度为0时停止计算
            # if vociety_field[i, j] < th * initial_velocity:
            #     break

    # 复原速度场
    if method == 'left':
            pass
    elif method == 'right':
        vociety_field = np.fliplr(vociety_field)  # 左右翻转图像
    elif method == 'top':
        vociety_field = vociety_field.T      # 转置图像
    elif method == 'bottom':
        vociety_field = np.fliplr(vociety_field).T  # 左右翻转并转置图像
    
    return vociety_field

# /-----------------------------------------------------------------------------------------------
# 参数定义
initial_velocity    = 255            # 初速度 (m/s)
density             = 50           # 物体密度 (kg/m^3)
area                = 9             # 面积 (m^2)
acceleration        = 40             # 加速度 (m/s^2)
dt                  = 1             # 时间间隔 (s) 

# /-----------------------------------------------------------------------------------------------------------------------------------/
# 图像获取
# img_gray, img_gray_noise = gaussian_draw() 
# img_gray = cv2.imread('D:\\6_Graduate\\DIC\\image_test\\Img000000_1.tif')
img_gray = cv2.imread('D:\\6_Graduate\\DIC\\image_test\\1.jpg')
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (512, 512))

# 计算摩擦场
friction = friction_calculate(img_gray).astype(np.uint8)
# sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
# friction = np.sqrt(sobel_x**2 + sobel_y**2)
friction = cv2.GaussianBlur(friction, (5, 5), 0)

# 计算速度场
vociety_field = vociety_calculate(img_gray, friction, initial_velocity, acceleration, method="left")
vociety_field = vociety_field.astype(np.uint8)
img_binary = cv2.threshold(vociety_field, 0, 255, cv2.THRESH_OTSU)[1]

# /--------------------------------------------------------------------------------------------------------------------------------/
# 显示结果
plt.subplot(131).imshow(img_gray, cmap='gray')
plt.title('Friction Image')
plt.subplot(132).imshow(friction, cmap='gray')
plt.title('Friction Image')
plt.subplot(133).imshow(vociety_field, cmap='gray')
plt.title('Friction Image')
plt.show()