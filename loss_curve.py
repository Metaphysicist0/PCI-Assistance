import matplotlib.pyplot as plt
import numpy as np

# 导入 matplotlib.pyplot
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# 设置字体大小，例如 22
plt.rcParams["font.size"] = 22
# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)


if __name__ == "__main__":
    train_loss_path = r"./diceloss.txt"  # 存储文件路径

    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('times')  # x轴标签
plt.ylabel('Loss')  # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="Dice")
plt.legend()
plt.title('DICE curve of RetineXNet-ResUNet++')
plt.show()


