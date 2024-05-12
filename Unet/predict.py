import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
from erzhi_to_touming import remove_zeros

def ce(test_path):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))

    # 测试模式
    net.eval()

    # 读取所有图片路径
    # tests_path = glob.glob('C:/Users/successful/Downloads/6/A0053.jpg')
    # # 遍历素有图片
    # for test_path in tests_path:



    # 加载原图
    original_image = cv2.imread(test_path)

    # 保存结果地址
    save_res_path = test_path.split('.')[0] + '_res.png'
    save_res_path1 = test_path.split('.')[0] + '_rees.png'

    # 读取图片
    img = cv2.imread(test_path)
    origin_shape = img.shape

    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (512, 512))

    # 转为batch为1，通道为1，大小为512*512的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])

    # 转为tensor
    img_tensor = torch.from_numpy(img)

    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    # 预测
    pred = net(img_tensor)

    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]

    # 处理结果
    pred[pred >= 0.9] = 255  # 设置置信度
    pred[pred < 0.9] = 0

    # 保存图片
    pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

    # 调整掩码图像的数据类型和大小
    mask = cv2.resize(pred, (original_image.shape[1], original_image.shape[0]))
    mask = mask.astype(np.uint8)

    # 生成掩码
    mask = cv2.bitwise_and(original_image, original_image, mask=mask)

    # 寻找最小外接矩形
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # 裁剪最小外接矩形
    mask_cropped = mask[y:y + h, x:x + w]

    # 设置尺寸大小
    size = (150, 150)
    mask_resized = cv2.resize(mask_cropped, size)

    # 保存裁剪并调整大小后的结果
    cv2.imwrite(save_res_path, mask_resized)

# test_path = 'C:/Users/successful/Downloads/6/ga1.png'
# ce(test_path)