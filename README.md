# ﻿云端目标检测与舵机云台系统

本工程当前采用“OrangePi采集摄像头画面 -> WebSocket发送到云端/本机服务 -> YOLO检测 -> Web页面展示 -> 可选舵机云台跟踪”的方案。当前已验证可运行的入口是三个脚本：cloud_linux.py、cloud_web.py、main_cloud.py。

## 当前可用功能

- cloud_linux.py：检测服务端。接收OrangePi发来的图像，调用YOLO模型推理，返回检测框、类别、置信度，并把最新画面和统计数据写入shared_data。
- cloud_web.py：Web可视化页面。读取shared_data，展示实时画面、系统状态、网络统计、检测结果列表。页面中保留本小组成员信息。
- main_cloud.py：OrangePi客户端。读取USB摄像头画面，发送给cloud_linux.py，并把返回的目标框交给舵机进程。
- cloud_client.py：WebSocket通信客户端，被main_cloud.py调用。
- servo.py、Adafruit_PCA9685/：舵机云台控制。检测到目标后，根据目标中心和画面中心的偏移控制PCA9685两路PWM。
- yolo11n.pt：YOLO通用预训练权重，可用于先验证流程。最终演示建议替换为自己训练的best.pt。
- cloud_linux_rknn.py：RKNN/NPU版本检测服务端，用yolo11n_fp.rknn在OrangePi NPU上推理。
- yolo11n.onnx、yolo11n_fp.rknn：由yolo11n.pt导出ONNX后，在本地WSL中使用rknn-toolkit2转换得到。
  
  
## 运行方式

在OrangePi上进入项目目录和环境：

cd ~/data0605/synthesisB_yolo_servo_detection
conda activate servo_new

终端1启动YOLO检测服务：

python cloud_linux.py --host 0.0.0.0 --port 8765 --model yolo11n.pt --data-dir ./shared_data --save-interval 0.3

使用自己训练的模型时，把--model yolo11n.pt换成：

--model runs/detect/custom_target/weights/best.pt

终端2启动Web页面：

python cloud_web.py --host 0.0.0.0 --port 8080 --data-dir ./shared_data --cache-time 0.2

浏览器访问：

http://172.20.10.2:8080

终端3启动OrangePi摄像头发送和云台跟踪：

python main_cloud.py --ws-url ws://127.0.0.1:8765

## RKNN实现简要说明

为了降低OrangePi上直接运行PyTorch/Ultralytics模型的负担，本项目增加了RKNN版本。转换流程为：

yolo11n.pt -> yolo11n.onnx -> yolo11n_fp.rknn

其中yolo11n.onnx由Rockchip适配的YOLO11导出方式生成，yolo11n_fp.rknn在本地WSL中使用rknn-toolkit2转换得到。OrangePi端只需要安装rknn-toolkit-lite2，并通过cloud_linux_rknn.py加载rknn模型进行NPU推理。Web页面、摄像头客户端、shared_data数据共享和舵机控制流程保持不变。

使用RKNN版本时，终端1改为启动：

python cloud_linux_rknn.py --host 0.0.0.0 --port 8765 --model yolo11n_fp.rknn --data-dir ./shared_data --save-interval 0.3

终端2和终端3仍然使用原来的cloud_web.py和main_cloud.py命令。

## 5x5卷积 whl 模块

my_wheel/保留了一个可复用的C++/pybind11 5x5卷积模块

在OrangePi上构建和测速：

cd ~/data0605/synthesisB_yolo_servo_detection/my_wheel
python -m pip install pybind11 wheel numpy
python setup.py bdist_wheel
python -m pip install dist/image_filter-*.whl --force-reinstall
python benchmark_filter.py --height 120 --width 160 --repeat 3
