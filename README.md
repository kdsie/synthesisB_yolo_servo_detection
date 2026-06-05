# 云端目标检测与舵机云台系统

小组成员：洪钰淇、丁宁、任筱舒

本工程当前采用“OrangePi采集摄像头画面 -> WebSocket发送到云端/本机服务 -> YOLO检测 -> Web页面展示 -> 可选舵机云台跟踪”的方案。当前已验证可运行的入口是三个脚本：`cloud_linux.py`、`cloud_web.py`、`main_cloud.py`。

## 当前可用功能

- `cloud_linux.py`：检测服务端。接收OrangePi发来的图像，调用YOLO模型推理，返回检测框、类别、置信度，并把最新画面和统计数据写入`shared_data`。
- `cloud_web.py`：Web可视化页面。读取`shared_data`，展示实时画面、系统状态、网络统计、检测结果列表。页面中保留本小组成员信息。
- `main_cloud.py`：OrangePi客户端。读取USB摄像头画面，发送给`cloud_linux.py`，并把返回的目标框交给舵机进程。
- `cloud_client.py`：WebSocket通信客户端，被`main_cloud.py`调用。
- `servo.py`、`Adafruit_PCA9685/`：舵机云台控制。检测到目标后，根据目标中心和画面中心的偏移控制PCA9685两路PWM。
- `yolo11n.pt`：YOLO通用预训练权重，可用于先验证流程。最终演示建议替换为自己训练的`best.pt`。

## 已清理或不再使用的功能

当前方案不再依赖RKNN/NPU本地推理，因此旧的`main.py`、RKNN模型、ONNX转换脚本、CPU本地检测脚本、旧备份、旧测试代码都可以移除。OrangePi当前系统缺少`/dev/rknpu`时，RKNN路线会报`failed to open rknpu module`，云端方案不受这个问题影响。

## 运行方式

在OrangePi上进入项目目录和环境：

```bash
cd ~/code0604
conda activate servo_new
```

终端1启动YOLO检测服务：

```bash
python cloud_linux.py --host 0.0.0.0 --port 8765 --model yolo11n.pt --data-dir ./shared_data --save-interval 0.3
```

使用自己训练的模型时，把`--model yolo11n.pt`换成：

```bash
--model runs/detect/custom_target/weights/best.pt
```

终端2启动Web页面：

```bash
python cloud_web.py --host 0.0.0.0 --port 8080 --data-dir ./shared_data --cache-time 0.2
```

浏览器访问：

```text
http://172.20.10.2:8080
```

终端3启动OrangePi摄像头发送和云台跟踪：

```bash
python main_cloud.py --ws-url ws://127.0.0.1:8765
```

如果只想展示检测，不接舵机：

```bash
python main_cloud.py --ws-url ws://127.0.0.1:8765 --no-servo
```

如果模型里有多个类别，只想让云台跟踪指定类别：

```bash
python main_cloud.py --ws-url ws://127.0.0.1:8765 --target-label your_target_name
```

## 为什么现在舵机云台看起来没有用上

舵机已经在`main_cloud.py`里启动，日志出现`进入servo_process`和`PCA9685初始化成功`说明控制进程已运行。它没有明显动作通常有几个原因：

- 当前使用`yolo11n.pt`通用COCO模型，检测结果是`person`、`chair`、`cup`等，云台会跟踪最高置信度目标，不一定是你真正想追踪的物体。
- 目标中心距离画面中心小于25像素时，代码认为无需调整。
- PCA9685的I2C、舵机供电、通道号、PWM范围不匹配时，软件有检测结果但硬件不动作。
- 如果运行时加了`--no-servo`，只会检测和展示，不会控制云台。

建议最终训练一个单类别模型，类别名就是你的目标名，然后用`--target-label 类别名`运行。这样检测、页面展示、舵机跟踪会围绕同一个特定目标。

## 自训练特定目标YOLO模型流程

建议检测目标选择：外观稳定、边界清楚、课堂现场容易反复摆放的物体。例如指定水杯、钥匙扣、实验板、某个包装盒、红色标志牌。不要选过小、反光强、和背景颜色太接近、经常被手遮挡的目标。

数据采集建议：

- 单类别目标至少采集200到500张图片；想更稳，采集800张以上。
- 包含近/中/远距离、亮/暗光、正面/侧面/倾斜、遮挡、不同背景。
- 额外拍一些没有目标的背景图，可降低误检。
- 图片分辨率不用太大，训练和部署统一使用640尺寸即可。

目录结构：

```text
datasets/custom/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

用LabelImg、CVAT或Roboflow标注矩形框，导出YOLO格式。每张图片对应一个同名`.txt`标签文件，格式为：

```text
class_id x_center y_center width height
```

坐标是0到1之间的归一化值。单类别时`class_id`为`0`。

编辑`datasets.yaml`：

```yaml
path: datasets/custom
train: images/train
val: images/val
nc: 1
names:
  0: your_target_name
```

安装训练依赖：

```bash
python -m pip install ultralytics opencv-python numpy websocket-client websockets flask pillow
```

开始训练：

```bash
yolo detect train model=yolo11n.pt data=datasets.yaml imgsz=640 epochs=100 batch=8 project=runs/detect name=custom_target
```

训练完成后权重在：

```text
runs/detect/custom_target/weights/best.pt
```

先离线测试：

```bash
yolo detect predict model=runs/detect/custom_target/weights/best.pt source=datasets/custom/images/val imgsz=640 conf=0.5
```

再替换到云端检测服务：

```bash
python cloud_linux.py --host 0.0.0.0 --port 8765 --model runs/detect/custom_target/weights/best.pt --data-dir ./shared_data --save-interval 0.3
```

最后启动客户端并指定跟踪目标：

```bash
python main_cloud.py --ws-url ws://127.0.0.1:8765 --target-label your_target_name
```

## 文件保留建议

最终提交和演示主要保留这些文件：

- `cloud_linux.py`
- `cloud_web.py`
- `cloud_client.py`
- `main_cloud.py`
- `servo.py`
- `Adafruit_PCA9685/`
- `datasets.yaml`
- `requirements.txt`
- `yolo11n.pt`
- 自己训练得到的`best.pt`
- `README.md`或`ReadMe_cloud.md`

## 5x5卷积 whl 加分模块

`my_wheel/`保留了一个可复用的C++/pybind11 5x5卷积模块，用于大作业加分项。它不影响YOLO检测主流程，但可以在报告中作为“C/C++实现小功能块并编译为whl，与Python实现对比性能”的内容。

在OrangePi上构建和测速：

```bash
cd ~/code0604/my_wheel
python -m pip install pybind11 wheel numpy
python setup.py bdist_wheel
python -m pip install dist/image_filter-*.whl --force-reinstall
python benchmark_filter.py --height 120 --width 160 --repeat 3
```