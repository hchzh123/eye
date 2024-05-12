
from PIL import Image

def remove_zeros(image_path, output_path):
    # 打开图像
    image = Image.open(image_path)

    # 转换为RGBA模式，方便处理透明度
    image = image.convert("RGBA")

    # 获取像素数据
    data = image.getdata()

    # 创建一个新的像素列表，将非零像素复制过去
    new_data = []
    for item in data:
        # 判断像素的R、G、B通道是否都为0
        if item[0] != 0 or item[1] != 0 or item[2] != 0:
            new_data.append(item)
        else:
            # 如果是0像素，设置透明度为0
            new_data.append((0, 0, 0, 0))

    # 创建新图像并保存
    new_image = Image.new("RGBA", image.size)
    new_image.putdata(new_data)
    new_image.save(output_path, "PNG")

# #
# image_path = R'C:\Users\successful\Downloads\6\A0053_res.png'
# output_path = R'C:\Users\successful\Downloads\6\A0053_res1.png'
#
# remove_zeros(image_path, output_path)
