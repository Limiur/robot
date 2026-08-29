# robot 控制算法项目
## 环境依赖
- Python 3.12
- numpy / control / matplotlib

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
## 文件夹说明
### DynamicProgramming
一份手写编辑的动态规划代码和一份用deepseek生成的代码
### LinearQuadraticRegulator
使用Copilot Agent生成的质量弹簧阻尼模型下是否使用LQR的对比  
补充一个有限时域内的LQR问题，代码暂时空缺