#这个是用来将红色掩膜转化为黑白的，用来减少数据量，提高分割速度
from PIL import Image
import os
import colorsys

# 定义图像文件夹路径和目标文件夹路径
image_folder = 'G:\\re_vessel590\\re_vessel590\\update_labelPNG590\\'
target_folder = 'G:\\re_vessel590\\re_vessel590\\output'

# 遍历图像文件夹中的所有文件
for filename in os.listdir(image_folder):
    # 获取文件路径
    filepath = os.path.join(image_folder, filename)

    # 打开图像文件
    image = Image.open(filepath)

    # 将图像转换为RGBA格式，便于处理透明度
    image = image.convert('RGBA')

    # 获取图像的像素数据
    data = image.load()

    # 遍历每个像素，将暗红色像素替换成白色像素
    for x in range(image.width):
        for y in range(image.height):
            r, g, b, a = data[x, y]
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            if h > 0.95 or h < 0.05:
                if s > 0.5 and v > 0.2:
                    data[x, y] = (255, 255, 255, a)

    # 将处理后的图像保存到目标文件夹
    target_filepath = os.path.join(target_folder, filename)
    image.save(target_filepath)
