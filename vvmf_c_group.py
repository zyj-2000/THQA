import pandas as pd
import math
import csv
import numpy as np


def get_points(n):
    """
    用户输入n,输出从y=e^(-x)函数中的n个(x,y)
    使得相邻两个x之间的距离相等,且所有y值相加和为1
    """
    # 定义y=e^(-x)函数
    def f(x):
        return math.exp(-1*x)

    # 计算所有y值之和
    total = 0
    for i in range(n):
        total += f(i)

    # 计算x的起始值和终止值
    x_start = 0
    x_end = n - 1

    # 计算相邻x之间的距离
    dx = (x_end - x_start) / (n - 1)

    # 初始化结果列表
    points = []

    # 遍历x值,计算相应的y值
    x = x_start
    for i in range(n):
        y = f(x) / total
        points.append((x, y))
        x += dx

    return points

def csv_write(subject, predict):
    with open('./vvmf_thzs_a1.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        predict = predict.tolist()
        predict.insert(0, subject)  # 在predict数组的第一个位置插入subject
        rows = [predict]
        print(rows)
        writer.writerows(rows)

def process(previous_image,png_list,predict_list):
    num = len(png_list)
    scores = []
    for jj in range(len(predict_list)):
        if num == 0:
            pass
        else:
            score = 0
            points = get_points(num)
            weights = []
            for i in range(num):
                weights.append(points[i][1])
            for j in range(num):
                score = score + predict_list[j] * weights[num-1-j]
            scores.append(score)
        # print(previous_image,scores)
        # csv_write(previous_image,score)

def process_g(previous_image,png_list,predict_list):
    num = len(png_list)
    scores = []

    if num == 0:
        pass
    else:
        score = np.zeros(len(predict_list[0]))
        points = get_points(num)
        weights = []
        for i in range(num):
            weights.append(points[i][1])
        for j in range(num):
            # print(predict_list[j])
            score = score + np.array(predict_list[j]) * weights[num-1-j]
           # print(score.shape)
        scores.append(score.tolist())
    # print(previous_image,len(scores))
        csv_write(previous_image,score)


# 读取test.csv文件
df = pd.read_csv('lib_c.csv')

# 初始化变量
previous_image = None
png_list = []
predict_list = []
# 逐行读取并处理数据
for index, row in df.iterrows():
    current_image = row['Image']
    current_png = row['png']
    # current_predict = row[2:28]
    current_predict = np.array(row[2:28]).tolist() # 将current_predict转换为列表形式

    # 判断Image列的内容是否和前一个不一样
    if current_image != previous_image:
        print("----------------------------")
        process_g(previous_image,png_list,predict_list)
        # # 如果和前一个不一样，新建png_list和predict_list
        # # print(png_list,predict_list)
        png_list = []
        predict_list = []
        png_list.append(current_png)
        predict_list.append(current_predict)
        previous_image = current_image
    else:
        # 如果和前一个一样，继续将该行的png和predict放入对应数组中
        png_list.append(current_png)
        predict_list.append(current_predict)
        previous_image = current_image
process_g(previous_image,png_list,predict_list)