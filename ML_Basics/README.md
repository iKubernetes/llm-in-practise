# 机器学习基础

- Python编程基础.ipynb：Python编程简单示例；
- NumPy示例.ipynb：NumPy实践示例；
- Matplotlib.ipynb：Matplotlib实践示例；
- Pandas编程基础.ipynb：Pandas实践示例；



## Matplotlib中文乱码问题解决

- 系统环境：Ubuntu Linux 22.04/24.04

### 安装字体

步骤一：安装中文字体库：

```bash
# 安装常用中文字体
sudo apt install fonts-noto-cjk fonts-wqy-microhei fonts-wqy-zenhei

# 安装微软雅黑字体（可选）
sudo apt install ttf-mscorefonts-installer -y
```

步骤二：确认字体安装成功

```bash
# 查看系统已安装中文字体
fc-list :lang=zh
```

该命令应该显示出类似如下内容：

```
/usr/share/fonts/truetype/wqy/wqy-microhei.ttc: WenQuanYi Micro Hei,文泉驛微米黑,文泉驿微米黑:style=Regular
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc: Noto Sans CJK SC:style=Regular
/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc: WenQuanYi Zen Hei,文泉驛正黑,文泉驿正黑:style=Regular
/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc: WenQuanYi Zen Hei Sharp,文泉驛點陣正黑,文泉驿点阵正黑:style=Regular
……
```

步骤三：更新Matploglib的字体缓存

```bash
# 删除缓存目录
rm -rf ~/.cache/matplotlib  

# 刷新系统字体缓存
fc-cache -fv  
```

### 配置Matplot使用中文字体

首先，使用类似如下代码打印Matplotlib支持的字体。

```bash
from matplotlib import pyplot as plt
from matplotlib import font_manager
from itertools import groupby

# 字体管理器
fm = font_manager.FontManager()
# 打印matplotlib支持的中文字体
print([f.name for f in fontManager.ttflist if 'CJK' in f.name or 'Hei' in f.name])
```

其输出内容应该包含类似如下行，这些就是Matplotlib支持的中文字体。

```
['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'Noto Serif CJK JP', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'Noto Serif CJK JP']
```

随后，从上面打印的字体列表中选择一个中文字体作为Matplotlib使用的字体即可。

#### 方法一：代码中动态设置（推荐）

```python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定中文字体（根据系统实际字体名选择）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 使用示例
plt.title('中文标题示例')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.plot([1,2,3], [4,5,6])
plt.show()
```

如果上面的代码能正常显示中文信息，则表示配置成功。

#### 方法二：修改配置文件（永久生效）

```bash
# 定位配置文件路径
python -c "import matplotlib; print(matplotlib.matplotlib_fname())"
# 输出路径类似于：/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/matplotlibrc

# 编辑配置文件
sudo vim 上述路径
```

找到并修改：

```yaml
#font.family:  sans-serif
font.family : Noto Sans CJK JP, WenQuanYi Micro Hei, WenQuanYi Zen Hei, sans-serif

#axes.unicode_minus: True
axes.unicode_minus: False
```

### 注意事项

**问题1：JupyterLab中不生效**

需要重启JupyterLab的内核，在JupyterLab的“内核”菜单中找到“重启内核”即可。

**问题2：特定符号无法显示**

```bash
sudo apt install fonts-noto-cjk-extra
```









