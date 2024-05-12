import cv2
import json
import numpy as np

# 加载图像

image = cv2.imread('C:/Users/successful/Desktop/ttys/87.png')#C:/Users/successful/Desktop/ttys/87.png

# 加载JSON并获取标注信息
with open('C:/Users/successful/Desktop/ttys/87.json', 'r') as f:
    data = json.load(f)
shapes = data['shapes']

# 创建一个与图像大小相同的掩码
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# 对每个标注物体进行处理
for i, shape in enumerate(shapes):
    label = shape['label']
    points = np.array(shape['points'], dtype=np.int32)

    # 在掩码上绘制多边形区域
    cv2.fillPoly(mask, [points], 255)

    # 根据掩码对原图进行分割
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # 创建带有透明通道的空白图像
    transparent_img = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # 将分割图像复制到透明图像的RGB通道中
    transparent_img[:, :, :3] = segmented_image

    # 创建透明度通道，将掩码中非零像素的值设置为255
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    alpha_channel[mask != 0] = 255

    # 将透明度通道添加到透明图像中
    transparent_img[:, :, 3] = alpha_channel

    # 保存带有透明背景的分割图像
    cv2.imwrite(f'segmented_transparent_{i}.png', transparent_img)

    # 显示分割结果
    cv2.imshow(f'Segmented - {label}', segmented_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()