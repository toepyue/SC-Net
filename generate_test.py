import cv2
import numpy as np

# 1. 读取你的真实原图 (确保 real_source.jpg 在同级目录)
img = cv2.imread('real_source.jpg')
img = cv2.resize(img, (400, 400))

# 2. 设定在模型“舒适区”内的变换参数 (中高难度: 放大1.3倍，旋转30度)
angle = -35
scale = 1.3
tx, ty = 30, -25 # 稍微平移一点

# 3. 生成目标图
center = (200.0, 200.0)
matrix = cv2.getRotationMatrix2D(center, angle, scale)
matrix[0, 2] += tx
matrix[1, 2] += ty

target_img = cv2.warpAffine(img, matrix, (400, 400))

# 4. 保存为严格合规的 real_target.jpg
cv2.imwrite('real_target.jpg', target_img)
print("✅ 已成功基于 real_source.jpg 生成合规的 real_target.jpg")
print(f"当前变换 -> 旋转: {angle}°, 缩放: {scale}x")