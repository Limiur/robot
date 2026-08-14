# -*- coding: utf-8 -*-
"""
DynamicProgramming(AI).py
=========================
垂直上飞轨迹的「时间最优」动态规划求解器（在 DynamicProgramming.py 基础上重写）。

问题描述
--------
飞行器从高度 h_min 处静止（v=0）垂直上飞至 h_max，到顶后速度归零（v=0），
速度 v ∈ [v_min, v_max]，每段匀加速（速度线性变化），段加速度满足
acc_min <= a <= acc_max，目标为总飞行时间最短。

模型 / 与原版的关系
-------------------
为了与原版矩阵输出完全一致，本版本沿用原版的表述：高度轴从 h_max 到 h_min
（h_grid 降序，行 0 = h_max = 起点），DP 从顶部状态 (h_max, v_start) 出发
沿高度下降方向递推；段加速度按原版约定计算：

        段耗时   T = (h[i-1] - h[i]) / ((v_起 + v_终) / 2)
        段加速度 a = (v_起 - v_终) / T            （原版约定，约束 acc_min<=a<=acc_max）

由于段耗时只取决于高度差与平均速度，下降方向的可行速度序列与上升方向一一对应
（时间反演），因此该表述与原问题（垂直上飞）等价：最优速度序列、最优总时间与
原版输出一致，最优轨迹按上升顺序输出即 (0,0) -> (2,2) -> (4,3) -> (6,3) -> (8,3) -> (10,0)。

本版本相对原版修正/完善的内容
-----------------------------
1. 边界状态修正（关键）：
   - 起始行 J[0, j != v_start] 原版残留 0（意味着"顶部任意速度代价为 0"），
     本版本改为 inf / NaN —— 飞行器只能以 v_start 从顶部出发，无法到达；
   - 终端行 J[N_h, j != v_end] 原版残留 0，本版本改为 inf / NaN ——
     即用户指出的 (0,1)(0,2)(0,3) 不应存在（飞行器只上飞，不会以非零速度
     出现在起点高度）。
2. 不再为第一段/最后一段写死特殊循环（原版 loop1/loop2/loop3 把 N_h=5、N_v=3
   写死在代码里），对任意网格通用。
3. 新增最优轨迹回溯（back 指针），按上升顺序输出完整轨迹表。
4. 新增穷举交叉验证（bruteforce_reference），小规模下核对 DP 结果。
5. 消除除零 RuntimeWarning。

用法
----
    dp = VerticalTrajectoryDP()          # 默认与原版一致的网格
    t  = dp.solve()                      # 最小飞行时间（inf 表示无可行轨迹）
    dp.summary()                         # 打印 J / 加速度表 + 最优轨迹表
"""

import numpy as np


class VerticalTrajectoryDP:
    """垂直上飞时间最优问题的动态规划求解器。

    状态 (i, j)：位于高度节点 i（h[i]，i=0..N_h，h 从 h_max 降到 h_min）
    且速度恰为 v[j]（j=0..N_v）。
    记 J[i, j] = 从初始状态 (h_max, v_start) 到达 (h[i], v[j]) 的最小累计时间。
    递推：J[i, b] = min_a [ T(a -> b) + J[i-1, a] ]，T(a -> b) 为第 i 段由
    速度 v[a] 匀加速到 v[b]（原版加速度约定）并下降 h[i-1] -> h[i] 所需时间。
    """

    def __init__(self, h_max=10.0, h_min=0.0, N_h=5,
                 v_min=0.0, v_max=3.0, N_v=3,
                 acc_min=-3.0, acc_max=2.0,
                 v_start=0.0, v_end=0.0):
        """
        Parameters
        ----------
        h_max, h_min : float     高度范围（h_max > h_min），h_max 为起点
        N_h          : int       高度分段数（节点数 = N_h + 1）
        v_min, v_max : float     速度范围（v_min <= v_max）
        N_v          : int       速度分段数（节点数 = N_v + 1）
        acc_min, acc_max : float 加速度约束范围（acc_min <= acc_max）
        v_start      : float     初始速度（默认 0：顶部静止出发）
        v_end        : float     终端速度（默认 0：到达 h_min 时速度为 0）
        """
        if N_h < 1 or N_v < 1:
            raise ValueError("N_h 与 N_v 必须 >= 1")
        if h_max <= h_min or v_max < v_min or acc_max < acc_min:
            raise ValueError("参数不合法：需 h_max > h_min, v_max >= v_min, acc_max >= acc_min")

        self.N_h, self.N_v = N_h, N_v
        self.h_grid = np.linspace(h_max, h_min, N_h + 1)   # 高度轴（降序，行 0 = 顶部）
        self.v_grid = np.linspace(v_min, v_max, N_v + 1)   # 速度轴（升序）
        self.acc_min, self.acc_max = acc_min, acc_max

        # 初始 / 终端状态（吸附到最近网格点）
        self.i_start, self.i_end = 0, N_h
        self.j_start = int(np.argmin(np.abs(self.v_grid - v_start)))
        self.j_end = int(np.argmin(np.abs(self.v_grid - v_end)))
        if abs(self.v_grid[self.j_start] - v_start) > 1e-9:
            print(f"[提示] v_start={v_start} 不在速度网格上，已吸附到 {self.v_grid[self.j_start]}")
        if abs(self.v_grid[self.j_end] - v_end) > 1e-9:
            print(f"[提示] v_end={v_end} 不在速度网格上，已吸附到 {self.v_grid[self.j_end]}")

        # DP 表
        self.J = None        # J[i, j] = 最小累计时间（inf = 不可达）
        self.acc_opt = None  # acc_opt[i, j] = 进入 (i, j) 那一段的加速度（原版约定；NaN = 不可达）
        self.back = None     # back[i, j] = 进入 (i, j) 的前一速度索引 a*
        self._solved = False

    # ------------------------------------------------------------------ #
    # 转移代价矩阵（与原版完全一致）
    # ------------------------------------------------------------------ #
    def _segment_transitions(self, i):
        """第 i 段（高度节点 i-1 -> i，下降）所有 (v_a -> v_b) 转移的时间与加速度。

        Returns
        -------
        T   : (N_v+1, N_v+1)，T[a, b] = 由 v[a] 匀加速到 v[b] 并下降该段的耗时；
              不满足约束或 v_avg=0 时为 inf
        acc : (N_v+1, N_v+1)，段加速度（原版约定 a=(v_起 - v_终)/T）；不可行段为 NaN
        """
        h_diff = self.h_grid[i - 1] - self.h_grid[i]        # > 0（下降）
        v_a = self.v_grid.reshape(-1, 1)                    # 行：段起始速度
        v_b = self.v_grid.reshape(1, -1)                    # 列：段终止速度
        v_avg = (v_a + v_b) / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            T = h_diff / v_avg
            acc = (v_a - v_b) / T                           # 原版约定的加速度

        # 可行性：v_avg > 0 且加速度在约束范围内
        ok = np.isfinite(T) & (T > 0.0) & (acc >= self.acc_min) & (acc <= self.acc_max)
        T = np.where(ok, T, np.inf)
        acc = np.where(ok, acc, np.nan)
        return T, acc

    # ------------------------------------------------------------------ #
    # 动态规划
    # ------------------------------------------------------------------ #
    def solve(self):
        """执行 DP，返回最小飞行时间（无可行轨迹时返回 inf）。"""
        self.J = np.full((self.N_h + 1, self.N_v + 1), np.inf)
        self.acc_opt = np.full((self.N_h + 1, self.N_v + 1), np.nan)
        self.back = np.full((self.N_h + 1, self.N_v + 1), -1, dtype=int)

        # 起始行：只有 (h_max, v_start) 代价为 0，其余不可达（inf）
        self.J[self.i_start, self.j_start] = 0.0
        self.acc_opt[self.i_start, self.j_start] = 0.0

        # 中间各行：完整递推（与原版 loop1/loop2 等价，且更通用）
        for i in range(1, self.N_h):
            T, acc = self._segment_transitions(i)
            total = self.J[i - 1, :].reshape(-1, 1) + T     # total[a,b] = J[i-1,a] + T[a,b]
            best = np.argmin(total, axis=0)                 # 每列最优的起始速度索引 a*
            self.J[i, :] = total[best, np.arange(self.N_v + 1)]
            self.back[i, :] = best
            acc_best = acc[best, np.arange(self.N_v + 1)]
            self.acc_opt[i, :] = np.where(np.isfinite(self.J[i, :]), acc_best, np.nan)

        # 终端行：只能以 v_end 到达终点，(N_h, j != v_end) 不可达（inf/NaN）
        if self.N_h >= 1:
            T, acc = self._segment_transitions(self.N_h)
            total = self.J[self.N_h - 1, :] + T[:, self.j_end]
            best = int(np.argmin(total))
            self.J[self.N_h, self.j_end] = total[best]
            self.back[self.N_h, self.j_end] = best
            self.acc_opt[self.N_h, self.j_end] = acc[best, self.j_end]

        self._solved = True
        return float(self.J[self.i_end, self.j_end])

    # ------------------------------------------------------------------ #
    # 最优轨迹回溯（按上升顺序输出，与用户期望的 (0,0)...(10,0) 一致）
    # ------------------------------------------------------------------ #
    def trajectory(self):
        """返回最优轨迹字典（高度从小到大）；无可行轨迹时返回 None。

        Returns
        -------
        dict with keys:
            height       : 各高度节点的轨迹高度（N_h+1,，升序）
            velocity     : 各高度节点处的速度（N_h+1,）
            segment_time : 每段的耗时（N_h,）
            segment_acc  : 每段的加速度（物理定义 a=(v_终-v_起)/T，N_h,）
            cum_time     : 累计时间（N_h+1,），cum_time[-1] = 总时间
            total_time   : 总时间标量
        """
        if not self._solved:
            self.solve()
        if not np.isfinite(self.J[self.i_end, self.j_end]):
            return None

        i, j = self.i_end, self.j_end
        heights, vels = [self.h_grid[i]], [self.v_grid[j]]
        seg_times, seg_accs = [], []
        while i > self.i_start:
            a_prev = int(self.back[i, j])
            h_diff = self.h_grid[i - 1] - self.h_grid[i]
            v_avg = (self.v_grid[a_prev] + self.v_grid[j]) / 2.0
            T_seg = h_diff / v_avg
            # 上升段从低处 (i, j) 到高处 (i-1, a_prev)，物理加速度 a = (v_高 - v_低)/T
            seg_times.append(T_seg)
            seg_accs.append((self.v_grid[a_prev] - self.v_grid[j]) / T_seg)
            i, j = i - 1, a_prev
            heights.append(self.h_grid[i])
            vels.append(self.v_grid[j])

        # 回溯自底向上进行（i 递减、h 递增），顺序即上升顺序，无需再反转
        cum = np.concatenate(([0.0], np.cumsum(seg_times)))

        assert abs(cum[-1] - self.J[self.i_end, self.j_end]) < 1e-9, "轨迹回溯与 DP 表不一致!"
        return {
            "height": np.asarray(heights),
            "velocity": np.asarray(vels),
            "segment_time": np.asarray(seg_times),
            "segment_acc": np.asarray(seg_accs),
            "cum_time": cum,
            "total_time": float(cum[-1]),
        }

    # ------------------------------------------------------------------ #
    # 穷举交叉验证（仅适用于小规模网格）
    # ------------------------------------------------------------------ #
    def bruteforce_reference(self):
        """穷举所有速度路径求最小飞行时间，验证 DP 结果（与原版约定一致）。

        路径 = 每个高度节点取一个速度节点；首末速度固定为 v_start / v_end。
        路径数 = (N_v+1)^(N_h-1)，超过 200 万条时跳过。
        """
        nv = self.N_v + 1
        n_paths = nv ** (self.N_h - 1)
        if n_paths > 2_000_000:
            print("[跳过] 路径数过多，不执行穷举验证。")
            return None

        segs = [self._segment_transitions(i) for i in range(1, self.N_h + 1)]
        best = np.inf
        for code in range(nv ** (self.N_h - 1)):
            p = [self.j_start]
            c = code
            for _ in range(self.N_h - 1):
                p.append(c % nv)
                c //= nv
            p.append(self.j_end)
            t = 0.0
            for i in range(1, self.N_h + 1):
                T, _ = segs[i - 1]
                Tij = T[p[i - 1], p[i]]
                if not np.isfinite(Tij):
                    t = np.inf
                    break
                t += Tij
            best = min(best, t)
        return float(best)

    # ------------------------------------------------------------------ #
    # 结果展示
    # ------------------------------------------------------------------ #
    def summary(self):
        """打印 J（最小到达时间）表、最优段加速度表与最优轨迹表。"""
        if not self._solved:
            self.solve()

        print("=" * 66)
        print("J（各状态最小到达时间, inf = 不可达）  行 = 高度(降序), 列 = 速度")
        print("=" * 66)
        print("高度轴 h =", np.round(self.h_grid, 4))
        print("速度轴 v =", np.round(self.v_grid, 4))
        with np.printoptions(precision=4, suppress=True):
            print("J =\n", self.J)
            print("acc_opt（最优段加速度, 原版约定, NaN = 不可达）=\n", self.acc_opt)

        traj = self.trajectory()
        if traj is None:
            print("\n[结论] 在当前约束下不存在可行轨迹（最小时间为 inf）。")
            return

        print("-" * 66)
        print("最优轨迹（垂直上飞, 按高度从小到大）")
        print("-" * 66)
        print(f"{'节点':>4} {'高度':>8} {'速度':>8} {'段耗时':>10} {'段加速度(物理)':>14} {'累计时间':>10}")
        for k in range(self.N_h + 1):
            if k == 0:
                seg_t, seg_a = "-", "-"
            else:
                seg_t = f"{traj['segment_time'][k - 1]:.4f}"
                seg_a = f"{traj['segment_acc'][k - 1]:.4f}"
            print(f"{k:>4} {traj['height'][k]:>8.3f} {traj['velocity'][k]:>8.3f} "
                  f"{seg_t:>10} {seg_a:>14} {traj['cum_time'][k]:>10.4f}")
        print(f"\n最小飞行时间 = {traj['total_time']:.6f} s")

        # 约束自检：物理定义下所有段加速度都应落在 [acc_min, acc_max]
        ok = np.all((traj['segment_acc'] >= self.acc_min - 1e-9) &
                    (traj['segment_acc'] <= self.acc_max + 1e-9))
        print(f"加速度约束自检: 所有段加速度均在 [{self.acc_min}, {self.acc_max}] 内 -> "
              f"{'[通过]' if ok else '[失败]'}")


# ---------------------------------------------------------------------- #
# 主程序：复现原问题配置 + 穷举验证
# ---------------------------------------------------------------------- #
def main():
    dp = VerticalTrajectoryDP(h_max=10, h_min=0, N_h=5,
                              v_min=0, v_max=3, N_v=3,
                              acc_min=-3, acc_max=2,
                              v_start=0, v_end=0)
    t_new = dp.solve()
    dp.summary()

    # 穷举交叉验证（小网格，秒级完成）
    t_bf = dp.bruteforce_reference()
    if t_bf is not None:
        match = abs(t_bf - t_new) < 1e-9
        print(f"穷举验证：最小飞行时间 = {t_bf:.6f} s  ->  "
              f"{'与 DP 结果一致 [OK]' if match else '与 DP 结果不一致 [FAIL]（存在 bug）'}")


if __name__ == "__main__":
    main()
