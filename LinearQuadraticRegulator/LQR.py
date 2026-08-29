"""质量-阻尼-弹簧系统的离散 LQR 控制与状态响应对比。"""

import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct


# 设置中文字体，避免标题和坐标轴标签乱码。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def main():
	# 定义质量、阻尼和弹簧刚度，并建立系统的状态空间模型。
	mass, damping, spring = 1.0, 0.5, 1.0
	A = np.array([[0.0, 1.0], [-spring / mass, -damping / mass]])
	B = np.array([[0.0], [1.0 / mass]])
	# 输出位移和速度，且系统不包含直接传递项。
	system = ct.ss(A, B, np.eye(2), np.zeros((2, 1)))
	print("初始模型状态方程：x_dot = A x + B u")
	print("A =\n", A)
	print("B =\n", B)

	dt = 0.1  # 离散化时间步长为 0.1 秒。
	# 使用零阶保持器将连续系统离散化，便于数字控制器实现。
	discrete_system = ct.c2d(system, dt, method="zoh")
	print("离散模型状态方程：x[k + 1] = A_d x[k] + B_d u[k]")
	print("A_d =\n", discrete_system.A)
	print("B_d =\n", discrete_system.B)

	time = np.arange(0.0, 20.0, dt)
	x0 = np.array([1.0, 0.0])  # 初始位移为 1.0，初始速度为 0.0。

	# 使用离散系统在无输入条件下计算初始响应。
	response = ct.initial_response(discrete_system, T=time, X0=x0)
	states = response.x

	# 设计离散 LQR 控制器，并使用状态反馈逐步计算控制输入。
	Q = np.diag([10.0, 1.0])
	R = np.array([[0.1]])
	K, S, E = ct.dlqr(discrete_system, Q, R)
	print("离散 LQR 控制器增益矩阵 K =\n", K)
	print("离散 LQR 离散 Riccati 方程的解 S =\n", S)
	controlled_states = np.zeros((2, len(time)))
	controlled_states[:, 0] = x0
	controlled_inputs = np.zeros(len(time) - 1)
	for index in range(len(time) - 1):
		controlled_inputs[index] = (-K @ controlled_states[:, index]).item()
		controlled_states[:, index + 1] = (
			discrete_system.A @ controlled_states[:, index]
			+ discrete_system.B[:, 0] * controlled_inputs[index]
		)

	# 在同一张图中上下对比无输入和 LQR 控制后的状态响应。
	fig, (ax_uncontrolled, ax_controlled) = plt.subplots(2, 1, sharex=True)
	for axis, plotted_states, title in (
		(ax_uncontrolled, states, "无输入时的系统响应"),
		(ax_controlled, controlled_states, "LQR 控制后的系统响应"),
	):
		axis.plot(time, plotted_states[0], label="位移")
		axis.plot(time, plotted_states[1], label="速度", color="tab:orange")
		axis.set_ylabel("状态值")
		axis.set_title(title)
		axis.grid(True)
		axis.legend()
	ax_controlled.set_xlabel("时间 (s)")

	fig.tight_layout()
	image_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), "state_comparison.png"
	)
	plt.savefig(image_path, dpi=150, bbox_inches="tight")
	plt.show()



if __name__ == "__main__":
	main()
