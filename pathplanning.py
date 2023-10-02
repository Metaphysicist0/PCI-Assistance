from PIL import Image
import numpy
#-*-coding:utf-8-*-
from pylab import *
import copy
import numpy as np
import matplotlib.pyplot as plt

step = 2
image = Image.open("C:\\Users\\YTL\\OneDrive\\Desktop\\path0423\\group\\ZhuJuping006.png")  # 用PIL中的Image.open打开图像
image_arr = np.array(image)  # 转化成numpy数组
#image_arr = np.squeeze(image_arr[:, :, 0])
image_arr = np.squeeze(image_arr[:, :, 0])
print(np.shape(image_arr))

image_arr = np.flipud(image_arr)
map_grid = np.float32(image_arr)
map_grid = map_grid[::step, ::step]
map_grid1 = np.copy(map_grid)

map_grid[map_grid <= 100] = 0
map_grid[map_grid > 100] = 10

map_grid1[map_grid1 <= 60] = 0
map_grid1[map_grid1 > 60] = 10
# -------------------

map_lon=np.shape(map_grid)[1]
map_lat=np.shape(map_grid)[0]

begin_a, begin_b = 191, 70
print(map_grid[begin_a, begin_b])
map_grid[begin_a, begin_b] = 5
end_a, end_b = 129, 105
map_grid[end_a, end_b] = 7


class modifiedAStar(object):

    def __init__(self):
        # self.g = 0  # g初始化为0
        self.start = numpy.array([begin_a, begin_b])  # 起点坐标
        self.goal = numpy.array([end_a, end_b])  # 终点坐标
        self.open = numpy.array([[], [], [], [], [], []])  # 先创建一个空的open表, 记录坐标，方向，g值，f值
        self.closed = numpy.array([[], [], [], [], [], []])  # 先创建一个空的closed表
        self.best_path_array = numpy.array([[], []])  # 回溯路径表

    def h_value_tem(self, son_p):
        h = (son_p[0] - self.goal[0]) ** 2 + (son_p[1] - self.goal[1]) ** 2
        h = numpy.sqrt(h)  # 计算h
        return h

    def g_accumulation(self, son_point, father_point):

        g1 = father_point[0] - son_point[0]
        g2 = father_point[1] - son_point[1]

        g_pingfang = g1 ** 2 + g2 ** 2
        g = numpy.sqrt(g_pingfang) + father_point[4]  # 加上累计的g值

        return g

    def f_value_tem(self, son_p, father_p):
        f = self.g_accumulation(son_p, father_p) + self.h_value_tem(son_p)
        return f

    def child_point(self, x):
        for j in range(-1, 2, 1):
            for q in range(-1, 2, 1):

                if j == 0 and q == 0:  # 搜索到父节点去掉
                    continue
                m = [x[0] + j, x[1] + q]
              
                if m[0] < 0 or m[0] > map_lon or m[1] < 0 or m[1] > map_lat: 
                    continue

                if map_grid[int(m[0]), int(m[1])] == 0:  
                    continue

                record_g = self.g_accumulation(m, x)
                record_f = self.f_value_tem(m, x)  

                x_direction, y_direction = self.direction(x, m)  

                para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  

                a, index = self.judge_location(m, self.open)
                if a == 1:
                    # 说明open中已经存在这个点

                    if record_f <= self.open[5][index]:
                        self.open[5][index] = record_f
                        self.open[4][index] = record_g
                        self.open[3][index] = y_direction
                        self.open[2][index] = x_direction

                    continue

                # 在closed表中,则去掉搜索点
                b, index2 = self.judge_location(m, self.closed)
                if b == 1:

                    if record_f <= self.closed[5][index2]:
                        self.closed[5][index2] = record_f
                        self.closed[4][index2] = record_g
                        self.closed[3][index2] = y_direction
                        self.closed[2][index2] = x_direction
                        self.closed = numpy.delete(self.closed, index2, axis=1)
                        self.open = numpy.c_[self.open, para]
                    continue

                self.open = numpy.c_[self.open, para]  # 参数添加到open中
                # print(self.open)


    def judge_location(self, m, list_co):
        jud = 0
        index = 0
        for i in range(list_co.shape[1]):

            if m[0] == list_co[0, i] and m[1] == list_co[1, i]:

                jud = jud + 1

                index = i
                break
            else:
                jud = jud
        # if a != 0:
        #     continue
        return jud, index

    def direction(self, father_point, son_point):
        x = son_point[0] - father_point[0]
        y = son_point[1] - father_point[1]
        return x, y

    def path_backtrace(self):

        best_path = [end_a, end_b]  # 回溯路径的初始化
        self.best_path_array = numpy.array([[end_a], [end_b]])
        j = 0
        while j <= self.closed.shape[1]:
            for i in range(self.closed.shape[1]):
                if best_path[0] == self.closed[0][i] and best_path[1] == self.closed[1][i]:
                    x = self.closed[0][i] - self.closed[2][i]
                    y = self.closed[1][i] - self.closed[3][i]
                    best_path = [x, y]
                    self.best_path_array = numpy.c_[self.best_path_array, best_path]
                    break  # 如果已经找到，退出本轮循环，减少耗时
                else:
                    continue
            j = j + 1
        # return best_path_array

    def main(self):
        best = self.start 
        h0 = self.h_value_tem(best)
        init_open = [best[0], best[1], 0, 0, 0, h0] 
        self.open = numpy.column_stack((self.open, init_open)) 

        ite = 1 
        while ite <= 100000000000000000000000.00000:

            if self.open.shape[1] == 0:
                print('没有')
                return

            self.open = self.open.T[numpy.lexsort(self.open)].T 


            best = self.open[:, 0]

            self.closed = numpy.c_[self.closed, best]

            if best[0] == end_a and best[1] == end_b:  
                print('有了')
                return

            self.child_point(best)  
            # print(self.open)
            self.open = numpy.delete(self.open, 0, axis=1)  

            # print(self.open)

            ite = ite + 1


class MAP(object):

    def draw_init_map(self):
        plt.imshow(map_grid, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
        # plt.colorbar()
        xlim(-1, map_lon)  # 设置x轴范围
        ylim(-1, map_lat)  # 设置y轴范围
        my_x_ticks = numpy.arange(0, map_lon, 20)
        my_y_ticks = numpy.arange(0, map_lat, 20)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.grid(True)
        # plt.show()

    def draw_path_open(self, a):
        map_open = copy.deepcopy(map_grid)
        for i in range(a.closed.shape[1]):
            x = a.closed[:, i]

            map_open[int(x[0]), int(x[1])] = 1

        plt.imshow(map_open, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
        # plt.colorbar()
        xlim(-1, map_lon)  # 设置x轴范围
        ylim(-1, map_lat)  # 设置y轴范围
        my_x_ticks = numpy.arange(0, map_lon, 20)
        my_y_ticks = numpy.arange(0, map_lat, 20)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.grid(True)
        # plt.show()

    def draw_path_closed(self, a):

        # print('closed长度：')
        # print(a.closed.shape[1])
        map_closed = copy.deepcopy(map_grid1)
        for i in range(a.closed.shape[1]):
            x = a.closed[:, i]

            map_closed[int(x[0]), int(x[1])] = 5

        plt.imshow(map_closed, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
        # plt.colorbar()
        xlim(-1, map_lon)  # 设置x轴范围
        ylim(-1, map_lat)  # 设置y轴范围
        my_x_ticks = numpy.arange(0, map_lon, 20)
        my_y_ticks = numpy.arange(0, map_lat, 20)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.grid(True)
        # plt.show()

    def draw_direction_point(self, a):

        map_direction = copy.deepcopy(map_grid1)
        for i in range(a.best_path_array.shape[1]):
            x = a.best_path_array[:, i]
            map_direction[int(x[0]), int(x[1])] = 10

        plt.imshow(map_direction, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
        plt.plot(a.best_path_array[1, :], a.best_path_array[0, :], linewidth=5, color='r')

        xlim(-1, map_lon)  # 设置x轴范围
        ylim(-1, map_lat)  # 设置y轴范围
        my_x_ticks = numpy.arange(0, map_lon, 20)
        my_y_ticks = numpy.arange(0, map_lat, 20)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.grid(True)


    def draw_three_axes(self, a):

        plt.figure()
        ax4 = plt.subplot()
        plt.sca(ax4)
        self.draw_direction_point(a)
        plt.show()


if __name__ == '__main__':
    a1 = modifiedAStar()
    a1.main()
    a1.path_backtrace()
    m1 = MAP()
    m1.draw_three_axes(a1)
