"""质量-阻尼-弹簧系统无输入时的自动调节过程。"""

import numpy as np
import matplotlib.pyplot as plt
import control


# 设置中文字体，避免标题和坐标轴标签乱码。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def main():
	# 定义质量、阻尼和弹簧刚度，并建立系统的状态空间模型。
	mass, damping, spring = 1.0, 0.5, 1.0
	A = np.array([[0.0, 1.0], [-spring / mass, -damping / mass]])
	B = np.array([[0.0], [1.0 / mass]])
	# 输出位移和速度，且系统不包含直接传递项。
	system = control.ss(A, B, np.eye(2), np.zeros((2, 1)))
	print("初始模型状态方程：x_dot = A x + B u")
	print("A =\n", A)
	print("B =\n", B)

	dt = 0.01
	# 使用零阶保持器将连续系统离散化，便于数字控制器实现。
	discrete_system = control.c2d(system, dt, method="zoh")
	print("离散模型状态方程：x[k + 1] = A_d x[k] + B_d u[k]")
	print("A_d =\n", np.asarray(discrete_system.A))
	print("B_d =\n", np.asarray(discrete_system.B))

	time = np.arange(0.0, 20.0 + dt, dt)
	x0 = [1.0, 0.0]

	# 使用离散系统在无输入条件下计算初始响应。
	response = control.initial_response(discrete_system, T=time, X0=x0)
	states = np.asarray(response.states)

	# 基于离散模型设计LQR控制器，并使用状态反馈进行闭环控制。
	Q = np.diag([10.0, 1.0])
	R = np.array([[0.1]])
	K, _, _ = control.dlqr(discrete_system, Q, R)
	controlled_states = np.zeros((2, len(time)))
	controlled_states[:, 0] = x0
	controlled_inputs = np.zeros(len(time) - 1)
	for index in range(len(time) - 1):
		u = -K @ controlled_states[:, index]
		controlled_inputs[index] = u.item()
		controlled_states[:, index + 1] = (
			discrete_system.A @ controlled_states[:, index]
			+ discrete_system.B[:, 0] * controlled_inputs[index]
		)

	# 上方绘制无输入响应，下方绘制LQR控制后的响应。
	fig, (ax_uncontrolled, ax_controlled) = plt.subplots(2, 1, sharex=True)
	for axis, plotted_states, title in (
		(ax_uncontrolled, states, "无输入时的系统响应"),
		(ax_controlled, controlled_states, "LQR控制后的系统响应"),
	):
		axis.plot(time, plotted_states[0], label="位移")
		axis.plot(time, plotted_states[1], label="速度", color="tab:orange")
		axis.set_xlabel("时间 (s)")
		axis.set_ylabel("状态值")
		axis.set_title(title)
		axis.grid(True)
		axis.legend()

	fig.tight_layout()
	plt.show()



if __name__ == "__main__":
	main()
