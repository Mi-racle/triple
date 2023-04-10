import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# 标准曲线
x = np.linspace(0, 1, 256)
t = np.sin(2 * np.pi * x)


# 采样函数
def get_data(N):
    x_n = np.linspace(0, 1, N)
    t_n = np.sin(2 * np.pi * x_n) + np.random.normal(scale=0.15, size=N)  # add Gaussian Noise
    return x_n, t_n


# 绘制部分组件函数
def draw_ticks():
    plt.tick_params(labelsize=15)
    plt.xticks(np.linspace(0, 1, 2))
    plt.yticks(np.linspace(-1, 1, 3))
    plt.ylim(-1.5, 1.5)
    font = {'family': 'Times New Roman', 'size': 20}
    plt.xlabel('x', font)
    plt.ylabel('t', font, rotation='horizontal')


# 拟合函数（lamda默认为0，即无正则项）
def regress(_m, _n, x, x_n, t_n, lamda=0):
    print("-----------------------M=%d, N=%d-------------------------" % (_m, _n))
    order = np.arange(_m + 1)
    order = order[:, np.newaxis]
    e = np.tile(order, [1, _n])
    XT = np.power(x_n, e)
    X = np.transpose(XT)
    a = np.matmul(XT, X) + lamda * np.identity(_m + 1)  # X.T * X
    b = np.matmul(XT, t_n)  # X.T * T
    ws = np.linalg.solve(a, b)  # aW = b => (X.T * X) * W = X.T * T
    print(ws)
    e2 = np.tile(order, [1, x.shape[0]])
    XT2 = np.power(x, e2)
    pr = np.matmul(ws, XT2)
    return pr, ws


def draw_curve(truth, pred, _x, _t, _m, _n):
    plt.figure(2, figsize=(8, 5))
    if pred is not None:
        plt.plot(x, truth, 'g', x, pred, 'r', linewidth=2)
    else:
        plt.plot(x, truth, 'g', linewidth=2)
    plt.scatter(_x, _t, marker='o', edgecolors='b', s=100, linewidth=2)
    draw_ticks()
    plt.title(f'Figure 2 : M = {_m}, N = {_n}')
    if pred is not None:
        plt.text(0.8, 0.9, f'M = {_m}', style='italic')
    target_path = f'runs/exp{run_index}'
    plt.savefig(f'{target_path}/M{_m}N{_n}.png', dpi=400)
    plt.clf()


if not os.path.exists('runs'):
    os.mkdir('runs')
run_index = 0
while True:
    if not os.path.exists(f'runs/exp{run_index}'):
        os.mkdir(f'runs/exp{run_index}')
        break
    run_index += 1

csv_file = open(f'runs/exp{run_index}/result.csv', 'w', newline='')
csvwriter = csv.writer(csv_file)

m_list = [0, 1, 3, 5, 9]
sampled_n = 256
# 采样
x_10, t_10 = get_data(sampled_n)
csvwriter.writerow(x)
csvwriter.writerow(t)
csvwriter.writerow(t_10)
draw_curve(t, None, x_10, t_10, -1, -1)
for m in m_list:
    p, w = regress(m, sampled_n, x, x_10, t_10)
    csvwriter.writerow(w)
    csvwriter.writerow(p)
    draw_curve(t, p, x_10, t_10, m, sampled_n)
