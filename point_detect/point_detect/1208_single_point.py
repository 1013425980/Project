import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# /-----------------------------------------------------------------------------------------------
# 参数定义
initial_velocity    = 50            # 初速度 (m/s)
density             = 50           # 物体密度 (kg/m^3)
area                = 9             # 面积 (m^2)
acceleration        = 10             # 加速度 (m/s^2)
dt                  = 1             # 时间间隔 (s) 

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

# /-----------------------------------------------------------------------------------------------
img_gray, img_gray_noise = gaussian_draw() 
# img_gray = cv2.imread('D:\\6_Graduate\\DIC\\image_test\\0.jpg')
# img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.resize(img_gray, (512, 512))

vociety_field = np.zeros_like(img_gray).astype(np.float32)
vociety_field[:, 0] = initial_velocity
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        if j == 0:    # 每一行的初始点
            vociety_field[i, j] = initial_velocity
        elif j == img_gray.shape[1]-1:    # 每一行的结束点
            vociety_field[i, j] = vociety_field[i, j-1]
        else:
            gray_diff = abs(img_gray[i, j] - img_gray[i, j-1])
            vociety_field[i, j] = vociety_field[i, j-1] - gray_diff
        
        if vociety_field[i, j] == 0 or vociety_field[i, j]<0:
            vociety_field[i, j:] = 0
            break

vociety_field = vociety_field.astype(np.uint8)
# /-----------------------------------------------------------------------------------------------
img_binary = cv2.threshold(vociety_field, 0, 255, cv2.THRESH_OTSU)[1]

plt.subplot(131).imshow(img_gray, cmap='gray')
plt.title('Gaussian Image')
plt.subplot(132).imshow(vociety_field, cmap='gray')
plt.title('Vociety Field')
plt.subplot(133).imshow(img_binary, cmap='gray')
plt.title('Binary Image')
plt.show()
