# robot 控制算法项目
## 环境依赖
- Python 3.12
- numpy / scipy 数值矩阵运算库

## Windows本地部署
- 克隆仓库
```bash
git clone https://github.com/Limiur/robot.git
```
- 创建虚拟环境
```bash
cd robot
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
### DynamicProgramming.py
手写编辑的动态规划代码
```bash
python DynamicProgramming.py
```
### DynamicProgramming(AI).py
用deepseek生成了一份代码
```bash
python DynamicProgramming.py
```
