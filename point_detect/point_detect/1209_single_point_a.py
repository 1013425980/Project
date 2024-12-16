import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# /-----------------------------------------------------------------------------------------------
# 参数定义
initial_velocity    = 100            # 初速度 (m/s)
density             = 50           # 物体密度 (kg/m^3)
area                = 9             # 面积 (m^2)
acceleration        = 1             # 加速度 (m/s^2)
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

# 反射边界填充
def reflection_padding(img, pad_size):
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    return padded_img

# 计算摩擦系数friction
def friction_calculate(window, method='DYNAMIC_RANGE'):
    if method == 'DYNAMIC_RANGE':
        friction = np.max(window) - np.min(window)  # 动态范围表征
    elif method == 'STD':
        friction = np.std(window)  # 标准差表征

    return friction

# 计算速度场
def vociety_calculate(img_gray, initial_velocity, acceleration, method='left'):
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
    for i in range(i_start, i_end, i_step):
        for j in range(j_start+1, j_end, j_step):
            window = img_gray[i, j-1:j+2]       # 窗口大小为1x3
        
            friction = friction_calculate(window)  # 计算窗口动态范围表征摩擦力
            if vociety_field[i, j-1] < initial_velocity:
                vociety_field[i, j] = vociety_field[i, j-1] + (acceleration - friction) * dt
            else:
                vociety_field[i, j] = vociety_field[i, j-1] - friction * dt
            vociety_field[i, j] = np.clip(vociety_field[i, j], 0, initial_velocity)
        
            # 速度为0时停止计算
            if vociety_field[i, j] < th * initial_velocity:
                break

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
# main
img_gray, img_gray_noise = gaussian_draw() 
# img_gray = cv2.imread('D:\\6_Graduate\\DIC\\image_test\\0.jpg')
# img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.resize(img_gray, (512, 512))

# 计算速度场
vociety_field = vociety_calculate(img_gray, initial_velocity, acceleration, method="bottom")
vociety_field = vociety_field.astype(np.uint8)
img_binary = cv2.threshold(vociety_field, 0, 255, cv2.THRESH_OTSU)[1]

# 分别计算四个方向速度场
vociety_field_left = vociety_calculate(img_gray, initial_velocity, acceleration, "left")
vociety_field_left = vociety_field_left.astype(np.uint8)
img_binary_left = cv2.threshold(vociety_field_left, 0, 255, cv2.THRESH_OTSU)[1]

vociety_field_top = vociety_calculate(img_gray, initial_velocity, acceleration, "top")
vociety_field_top = vociety_field_top.astype(np.uint8)
img_binary_top = cv2.threshold(vociety_field_top, 0, 255, cv2.THRESH_OTSU)[1]

vociety_field_bottom = vociety_calculate(img_gray, initial_velocity, acceleration, "bottom")
vociety_field_bottom = vociety_field_bottom.astype(np.uint8)
img_binary_bottom = cv2.threshold(vociety_field_bottom, 0, 255, cv2.THRESH_OTSU)[1]

vociety_field_right = vociety_calculate(img_gray, initial_velocity, acceleration, "right")
vociety_field_right = vociety_field_right.astype(np.uint8)
img_binary_right = cv2.threshold(vociety_field_right, 0, 255, cv2.THRESH_OTSU)[1]

vociety_field_lr = np.zeros_like(vociety_field).astype(np.uint32)
vociety_field_lr = (vociety_field_left + vociety_field_right)
vociety_field_lr = vociety_field_lr.astype(np.uint8)
vociety_field_binary_lr = cv2.threshold(vociety_field_lr, 20, 255, cv2.THRESH_BINARY)[1]
vociety_field_tb = np.zeros_like(vociety_field).astype(np.uint32)
vociety_field_tb = (vociety_field_top + vociety_field_bottom)
vociety_field_tb = vociety_field_tb.astype(np.uint8)
vociety_field_binary_tb = cv2.threshold(vociety_field_tb, 20, 255, cv2.THRESH_BINARY)[1]

vociety_field_all = np.bitwise_and(vociety_field_binary_lr, vociety_field_binary_tb)

plt.subplot(231).imshow(vociety_field_left, cmap='gray')
plt.title('Binary Image Left')
plt.subplot(232).imshow(vociety_field_top, cmap='gray')
plt.title('Binary Image Top')
plt.subplot(233).imshow(vociety_field_lr, cmap='gray')
plt.title('Gray Image LR')
plt.subplot(234).imshow(vociety_field_tb, cmap='gray')
plt.title('Gray Image TB')
plt.subplot(236).imshow(vociety_field_binary_lr, cmap='gray')
plt.title('Binary Image LR')
plt.subplot(235).imshow(vociety_field_binary_tb, cmap='gray')
plt.title('Binary Image tb')
plt.subplot(236).imshow(vociety_field_all, cmap='gray')
plt.title('Binary Image LTRB')
plt.show()
