# robot 控制算法项目
## 环境依赖
- Python 3.12
- numpy / scipy 数值矩阵运算库

## Windows本地部署
- 克隆仓库
```bash
git clone https://github.com/Limiur/robot.git
cd robot
```
- 创建虚拟环境
```bash
python -m venv .venv
```
- 安装依赖
```bash
pip install -r requirements.txt
```
- 激活环境
```bash
.venv\Scripts\activate 
```
- 运行代码
```bash
python demo.py
```
## 新增模块说明
### lqr_solver.py
实现LQR最优控制器求解，输入状态空间矩阵A/B/Q/R，返回反馈增益矩阵K。

使用示例：
```bash
python lqr_solver.py
```
