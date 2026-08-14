import numpy as np
import scipy

# 设置高度范围和速度范围
h_max = 10; h_min = 0
v_max = 3; v_min = 0
# 设置高度和速度离散度
N_h = 5; N_v = 3
# 定义高度轴
h_col = np.linspace(h_max, h_min, N_h+1)
# 定义速度轴
v_row = np.linspace(v_min, v_max, N_v+1)
# 添加加速度约束
acc_max = 2; acc_min = -3
# 定义代价函数
J_costtogo = np.zeros((N_h+1, N_v+1))
# 定义加速度矩阵
acc_mat = np.zeros((N_h+1, N_v+1))


# 计算10->8的代价函数
def loop1():
    # 计算高度差、平均速度以及速度差
    h_diff = h_col[0] - h_col[1]
    v_avg = (v_row[0] + v_row) / 2
    v_diff = (v_row[0] - v_row)
    # 计算时间消耗
    T_delta = h_diff / v_avg
    acc_ = v_diff / T_delta
    J_temp = T_delta
    
    # 将不满足加速度约束的代价设置为无穷大
    mask_bad = (acc_ > acc_max) | (acc_ < acc_min)
    J_temp[mask_bad] = np.inf

    J_costtogo[1, :] = J_temp
    acc_mat[1, :] = acc_


# 计算8->6->4->2的代价函数
def loop2():
    for i in range(2, N_h):
        # 计算高度差和平均速度和速度差
        h_diff = h_col[i-1] - h_col[i]
        v_avg = (v_row.reshape(-1,1) + v_row) / 2
        v_diff = (v_row.reshape(-1,1) - v_row)
        # 当前代价等于时间消耗
        T_delta = h_diff / v_avg
        acc_ = v_diff / T_delta
        J_temp = T_delta

        # 将不满足加速度约束的代价设置为无穷大
        mask_bad = (acc_ > acc_max) | (acc_ < acc_min)
        J_temp[mask_bad] = np.inf
        J_temp = J_temp + (J_costtogo[i-1, :]).reshape(-1, 1)

        J_costtogo[i, :] = np.min(J_temp, axis=0)
        acc_mat[i, :] = acc_[np.argmin(J_temp, axis=0), np.arange(N_v+1)]


# 计算2->0的代价函数
def loop3():
    # 计算高度差和平均速度和速度差
    h_diff = h_col[4] - h_col[5]
    v_avg = (v_row + v_row[0]) / 2
    v_diff = (v_row - v_row[0])
    # 当前代价等于时间消耗
    T_delta = h_diff / v_avg
    acc_ = (v_diff / T_delta).reshape(-1, 1)
    J_temp = T_delta.reshape(-1, 1)

    # 将不满足加速度约束的代价设置为无穷大
    mask_bad = (acc_ > acc_max) | (acc_ < acc_min)
    J_temp[mask_bad] = np.inf
    J_temp = J_temp + (J_costtogo[4, :]).reshape(-1, 1)

    # 形状处理
    JJ = np.zeros((N_v+1, N_v+1))
    JJ[:, 0] = J_temp[:, 0]
    AA = np.zeros((N_v+1, N_v+1))
    AA[:, 0] = acc_[:, 0]

    J_costtogo[5, :] = np.min(JJ, axis=0)
    acc_mat[5, :] = AA[np.argmin(JJ, axis=0), np.arange(N_v+1)]


def main():
    loop1()
    loop2()
    loop3()
    print("J_costtogo:\n", J_costtogo)
    print("acc_mat:\n", acc_mat)


if __name__ == "__main__":
    main()
