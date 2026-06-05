#!/usr/bin/env python3
# WebSocket服务器: 接收图像并返回检测结果 (轻量级版) + Web界面展示

import os
import time
import json
import cv2
import numpy as np
import asyncio
import websockets
import base64
from ultralytics import YOLO
import argparse
import threading
import pickle

# 默认端口设置
DEFAULT_WS_PORT = 8765
DEFAULT_WEB_PORT = 8080

# 数据共享目录 - 用于与cloud_web.py共享数据
DATA_DIR = "./shared_data"
os.makedirs(DATA_DIR, exist_ok=True)

# 全局模型变量
model = None
# 添加处理计数器
processed_frames = 0
last_report_time = time.time()

# 网络延迟统计
network_stats = {
    "receive_times": [],  # 接收请求延迟
    "process_times": [],  # 处理时间
    "send_times": [],     # 发送响应延迟
    "total_times": [],    # 总处理时间
    "json_serialize_times": [],  # JSON序列化时间
    "send_start_times": [],      # 发送开始时间
    "send_complete_times": [],   # 发送完成时间
    "response_sizes": []         # 响应大小（字节）
}
stats_lock = threading.Lock()

# 存储最新的检测结果用于Web展示
latest_detections = []
latest_frame = None
latest_stats = {
    "fps": 0,
    "total_frames": 0,
    "avg_process_time": 0,
    "last_detection_time": "-"
}
web_data_lock = threading.Lock()

# 简化的内存共享变量，用于减少文件IO
shared_memory = {
    "latest_frame_jpg": None,  # JPEG编码的最新帧
    "last_save_time": 0,       # 上次保存到文件的时间
    "save_interval": 0.3       # 保存到文件的间隔，网页端依赖它刷新画面
}

# 添加历史距离记录和计算平均值的功能
distance_history = {
    "values": [],  # 存储最近的距离值
    "max_history": 20  # 最多保存20个历史记录
}

def save_data_to_file(filename, data):
    """将数据保存到文件 - 简化版，无锁"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"保存数据到文件失败: {e}")
        return False

def save_image_bytes(filename, img_bytes):
    """将图像字节数据保存到文件 - 简化版，无锁"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"保存图像到文件失败: {e}")
        return False

def print_network_stats():
    """打印网络统计信息"""
    with stats_lock:
        if not network_stats["total_times"]:
            print("尚无网络统计数据")
            return
            
        avg_receive = sum(network_stats["receive_times"]) / len(network_stats["receive_times"])
        avg_process = sum(network_stats["process_times"]) / len(network_stats["process_times"])
        avg_send = sum(network_stats["send_times"]) / len(network_stats["send_times"])
        avg_total = sum(network_stats["total_times"]) / len(network_stats["total_times"])
        
        # 新增统计
        if network_stats["json_serialize_times"]:
            avg_json = sum(network_stats["json_serialize_times"]) / len(network_stats["json_serialize_times"])
        else:
            avg_json = 0
            
        if network_stats["send_start_times"] and network_stats["send_complete_times"]:
            avg_send_start = sum(network_stats["send_start_times"]) / len(network_stats["send_start_times"])
            avg_send_complete = sum(network_stats["send_complete_times"]) / len(network_stats["send_complete_times"])
        else:
            avg_send_start = 0
            avg_send_complete = 0
            
        if network_stats["response_sizes"]:
            avg_size = sum(network_stats["response_sizes"]) / len(network_stats["response_sizes"])
            max_size = max(network_stats["response_sizes"])
            min_size = min(network_stats["response_sizes"])
        else:
            avg_size = 0
            max_size = 0
            min_size = 0
        
        max_total = max(network_stats["total_times"])
        min_total = min(network_stats["total_times"])
        
        print("\n===== 服务器端延迟统计 =====")
        print(f"接收请求延迟 (平均): {avg_receive:.1f}ms")
        print(f"处理时间 (平均): {avg_process:.1f}ms")
        print(f"JSON序列化时间 (平均): {avg_json:.1f}ms")
        print(f"发送开始前准备时间 (平均): {avg_send_start:.1f}ms")
        print(f"发送完成时间 (平均): {avg_send_complete:.1f}ms")
        print(f"总发送时间 (平均): {avg_send:.1f}ms")
        print(f"总处理时间 (平均): {avg_total:.1f}ms")
        print(f"总处理时间 (最大/最小): {max_total:.1f}ms / {min_total:.1f}ms")
        print(f"响应大小 (平均): {avg_size:.1f}字节 ({avg_size/1024:.2f}KB)")
        print(f"响应大小 (最大/最小): {max_size}字节 ({max_size/1024:.2f}KB) / {min_size}字节 ({min_size/1024:.2f}KB)")
        print("============================\n")
        
        # 更新Web展示的统计数据
        with web_data_lock:
            latest_stats["avg_process_time"] = avg_process
        
        # 计算理论传输时间
        bandwidth_mbps = 1.0  # 假设带宽为1Mbps
        bandwidth_kbps = bandwidth_mbps * 1024  # 转换为Kbps
        bandwidth_kBps = bandwidth_kbps / 8  # 转换为KB/s
        
        theoretical_time_ms = (avg_size / 1024) / bandwidth_kBps * 1000  # 毫秒
        
        print(f"理论传输时间 (1Mbps带宽): {theoretical_time_ms:.1f}ms")
        print(f"实际网络传输时间 (平均): {avg_send:.1f}ms")
        print("============================\n")

def save_state_periodically():
    """定期将内存中的状态保存到文件"""
    while True:
        try:
            current_time = time.time()
            
            # 每隔一段时间保存一次状态到文件
            if current_time - shared_memory["last_save_time"] >= shared_memory["save_interval"]:
                # 保存检测结果
                save_data_to_file("latest_detections.pkl", latest_detections)
                
                # 保存统计信息
                save_data_to_file("latest_stats.pkl", latest_stats)
                
                # 保存网络统计
                save_data_to_file("network_stats.pkl", network_stats)
                
                # 保存最新帧
                if shared_memory["latest_frame_jpg"] is not None:
                    save_image_bytes("latest_frame.jpg", shared_memory["latest_frame_jpg"])
                
                shared_memory["last_save_time"] = current_time
                print(f"状态已保存到文件，时间: {time.strftime('%H:%M:%S')}")
            
            # 按配置间隔检查，便于网页端接近实时刷新。
            time.sleep(max(0.05, min(shared_memory["save_interval"], 1.0)))
        except Exception as e:
            print(f"保存状态失败: {e}")
            time.sleep(5)  # 出错后等待更长时间

async def process_image(websocket, path=None):
    """处理WebSocket连接并处理图像"""
    global processed_frames, last_report_time, latest_detections, latest_frame, latest_stats
    
    client_address = getattr(websocket, 'remote_address', 'unknown')
    print(f"客户端已连接: {client_address}")
    
    frame_count = 0
    stats_timer = time.time()  # 用于定期打印网络统计
    
    try:
        async for message in websocket:
            # 记录接收开始时间
            receive_start = time.time()
            
            try:
                # 解析JSON消息
                data = json.loads(message)
                img_base64 = data.get('image')
                
                # 记录接收完成时间
                receive_end = time.time()
                receive_time = (receive_end - receive_start) * 1000  # 毫秒
                
                if not img_base64:
                    await websocket.send(json.dumps({"error": "未收到图像数据"}))
                    continue
                
                # 解码Base64图像 - 使用numpy直接解码更快
                decode_start = time.time()
                img_bytes = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                decode_time = (time.time() - decode_start) * 1000  # 毫秒
                
                if img is None:
                    await websocket.send(json.dumps({"error": "图像解码失败"}))
                    continue
                
                # 保存最新帧用于Web展示 - 直接存储在内存中
                with web_data_lock:
                    latest_frame = img.copy()
                    
                    # 压缩图像以减小大小，并保存到共享内存
                    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    shared_memory["latest_frame_jpg"] = img_encoded.tobytes()
                
                # 确保图像大小为640x640 - 如果已经是正确尺寸则跳过调整
                if img.shape[0] != 640 or img.shape[1] != 640:
                    img = cv2.resize(img, (640, 640))
                
                frame_count += 1
                processed_frames += 1
                
                # 每秒只打印一次性能统计，而不是每帧都打印
                current_time = time.time()
                if current_time - last_report_time >= 1.0:
                    fps = processed_frames / (current_time - last_report_time)
                    print(f"处理速度: {fps:.1f} FPS (总共处理了 {frame_count} 帧)")
                    processed_frames = 0
                    last_report_time = current_time
                    
                    # 更新Web展示的FPS
                    with web_data_lock:
                        latest_stats["fps"] = fps
                        latest_stats["total_frames"] = frame_count
                
                # 每10秒打印一次网络统计
                if current_time - stats_timer >= 10.0:
                    print_network_stats()
                    stats_timer = current_time
                
                # 使用YOLO模型进行预测
                inference_start = time.time()
                results = model(img, verbose=False)
                inference_time = time.time() - inference_start
                
                # 解析结果 - 只返回置信度大于0.5的检测结果
                detections = []
                confidence_threshold = 0.5  # 置信度阈值
                
                if len(results) > 0:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0].item())
                        class_id = int(box.cls[0].item())
                        label = results[0].names[class_id]
                        
                        # 只添加置信度高于阈值的检测结果
                        if confidence > confidence_threshold:
                            # 计算边界框中心坐标
                            box_center_x = (x1 + x2) / 2
                            box_center_y = (y1 + y2) / 2
                            
                            # 计算图像中心坐标 (假设图像大小为640x640)
                            image_center_x = img.shape[1] / 2
                            image_center_y = img.shape[0] / 2
                            
                            # 计算边界框中心到图像中心的距离
                            distance_to_center = np.sqrt((box_center_x - image_center_x)**2 + 
                                                        (box_center_y - image_center_y)**2)
                            
                            # 将距离添加到历史记录中
                            distance_history["values"].append(distance_to_center)
                            # 保持历史记录长度不超过最大值
                            if len(distance_history["values"]) > distance_history["max_history"]:
                                distance_history["values"] = distance_history["values"][-distance_history["max_history"]:]
                            
                            # 计算平均距离
                            avg_distance = sum(distance_history["values"]) / len(distance_history["values"]) if distance_history["values"] else 0
                            
                            detections.append({
                                'l': label,  # 缩短字段名
                                'c': round(confidence, 3),  # 缩短字段名并限制精度
                                'b': [int(x1), int(y1), int(x2), int(y2)],  # 缩短字段名
                                'd': round(distance_to_center, 1),  # 到中心的距离
                                'a': round(avg_distance, 1)  # 平均距离
                            })
                
                # 更新最新检测结果用于Web展示
                with web_data_lock:
                    latest_detections = detections
                    latest_stats["last_detection_time"] = time.strftime("%H:%M:%S", time.localtime())
                
                # 计算处理时间（从接收到解析结果）
                process_time = (time.time() - receive_end) * 1000  # 毫秒
                
                # 准备返回的结果 - 只包含必要信息，不包含图像数据
                result = {
                    "f": frame_count,  # 缩短字段名
                    "n": len(detections),  # 缩短字段名
                    "d": detections,  # 缩短字段名
                    "i": round(inference_time, 4),  # 缩短字段名并限制精度
                    "t": {  # 缩短字段名
                        "r": round(receive_time, 1),  # 缩短字段名并限制精度
                        "p": round(process_time, 1),  # 缩短字段名并限制精度
                        "t": round(receive_time + process_time, 1)  # 缩短字段名并限制精度
                    }
                }
                
                # 记录JSON序列化开始时间
                json_start = time.time()
                
                # 序列化JSON
                json_result = json.dumps(result)
                
                # 计算响应大小
                response_size = len(json_result.encode('utf-8'))
                
                # 记录JSON序列化时间
                json_time = (time.time() - json_start) * 1000  # 毫秒
                
                # 记录发送前准备时间
                send_prep_start = time.time()
                
                # 记录发送开始时间
                send_start = time.time()
                
                # 发送前准备时间
                send_prep_time = (send_start - send_prep_start) * 1000  # 毫秒
                
                # 发送响应
                await websocket.send(json_result)
                
                # 记录发送完成时间
                send_complete = time.time()
                send_complete_time = (send_complete - send_start) * 1000  # 毫秒
                
                # 记录总发送时间
                send_time = (send_complete - send_start) * 1000  # 毫秒
                
                # 计算总时间
                total_time = receive_time + process_time + send_time
                
                # 保存统计数据
                with stats_lock:
                    network_stats["receive_times"].append(receive_time)
                    network_stats["process_times"].append(process_time)
                    network_stats["send_times"].append(send_time)
                    network_stats["total_times"].append(total_time)
                    network_stats["json_serialize_times"].append(json_time)
                    network_stats["send_start_times"].append(send_prep_time)
                    network_stats["send_complete_times"].append(send_complete_time)
                    network_stats["response_sizes"].append(response_size)
                    
                    # 只保留最近50个样本
                    if len(network_stats["total_times"]) > 50:
                        network_stats["receive_times"] = network_stats["receive_times"][-50:]
                        network_stats["process_times"] = network_stats["process_times"][-50:]
                        network_stats["send_times"] = network_stats["send_times"][-50:]
                        network_stats["total_times"] = network_stats["total_times"][-50:]
                        network_stats["json_serialize_times"] = network_stats["json_serialize_times"][-50:]
                        network_stats["send_start_times"] = network_stats["send_start_times"][-50:]
                        network_stats["send_complete_times"] = network_stats["send_complete_times"][-50:]
                        network_stats["response_sizes"] = network_stats["response_sizes"][-50:]
                
                # 只在检测到物体时打印详细信息
                if len(detections) > 0:
                    print(f"第 {frame_count} 帧检测到 {len(detections)} 个对象")
                    print(f"接收: {receive_time:.1f}ms, 处理: {process_time:.1f}ms, JSON: {json_time:.1f}ms")
                    print(f"发送准备: {send_prep_time:.1f}ms, 发送完成: {send_complete_time:.1f}ms, 总发送: {send_time:.1f}ms")
                    print(f"响应大小: {response_size}字节 ({response_size/1024:.2f}KB)")
                    print(f"总计: {total_time:.1f}ms")
                    
                    # 计算理论传输时间
                    bandwidth_mbps = 1.0  # 假设带宽为1Mbps
                    bandwidth_kbps = bandwidth_mbps * 1024  # 转换为Kbps
                    bandwidth_kBps = bandwidth_kbps / 8  # 转换为KB/s
                    theoretical_time_ms = (response_size / 1024) / bandwidth_kBps * 1000  # 毫秒
                    
                    print(f"理论传输时间 (1Mbps带宽): {theoretical_time_ms:.1f}ms")
                
            except json.JSONDecodeError:
                print("JSON解析错误")
                await websocket.send(json.dumps({"error": "JSON解析错误"}))
            except Exception as e:
                print(f"处理图像时出错: {e}")
                await websocket.send(json.dumps({"error": str(e)}))
    
    except websockets.exceptions.ConnectionClosed:
        print(f"客户端断开连接: {client_address}")
        # 打印最终网络统计
        print_network_stats()
    except Exception as e:
        print(f"WebSocket错误: {e}")
        import traceback
        traceback.print_exc()

def load_model(model_path, use_onnx=True):
    global model
    
    try:
        print(f"正在加载模型: {model_path}")
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return False
            
        # 检查是否使用ONNX优化
        if use_onnx:
            # 获取模型名称（不含扩展名）
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            onnx_path = f"{model_name}_640_640.onnx"  # 使用640x640尺寸
            
            # 如果ONNX模型不存在，则创建
            if not os.path.exists(onnx_path):
                print(f"ONNX模型不存在，正在从{model_path}导出...")
                temp_model = YOLO(model_path)
                temp_model.export(format="onnx", imgsz=[640, 640], half=True)  # 使用640x640尺寸
                print(f"ONNX模型已导出: {onnx_path}")
                
            # 加载ONNX模型
            print(f"正在加载ONNX模型: {onnx_path}")
            model = YOLO(onnx_path)
            print("ONNX模型加载成功")
        else:
            # 加载普通PyTorch模型
            model = YOLO(model_path)
            print("模型加载成功")
            
        # 模型预热
        print("正在进行模型预热...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)  # 使用640x640尺寸
        for _ in range(5):  # 减少预热次数以加快启动
            _ = model(dummy_img, verbose=False)
        print("模型预热完成")
        
        return True
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False

async def run_websocket_server(host, port, model_path, use_onnx=True):
    """运行WebSocket服务器"""
    # 加载模型
    if not load_model(model_path, use_onnx):
        print("模型加载失败，服务器无法启动")
        return
    
    # 启动WebSocket服务器
    print(f"WebSocket服务器启动，监听 {host}:{port}")
    
    # 启动后台保存线程
    save_thread = threading.Thread(target=save_state_periodically, daemon=True)
    save_thread.start()
    
    # 增加最大消息大小和最大连接数，禁用压缩以提高速度
    async with websockets.serve(
        process_image, 
        host, 
        port, 
        max_size=10*1024*1024,  # 10MB最大消息大小
        max_queue=32,  # 增加队列大小
        compression=None,  # 禁用压缩以提高速度
        ping_interval=None,  # 禁用ping以减少开销
        ping_timeout=None
    ):
        print(f"WebSocket URL: ws://{host}:{port}")
        print("等待客户端连接...")
        await asyncio.Future()  # 运行直到被取消

def main():
    """主函数"""
    # 声明全局变量
    global DATA_DIR
    
    parser = argparse.ArgumentParser(description="WebSocket目标检测服务器 (轻量级版) + Web界面")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket监听地址，默认监听所有地址")
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT, help="WebSocket监听端口，默认8765")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--use-onnx", action="store_true", help="是否使用ONNX优化")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据共享目录路径")
    parser.add_argument("--save-interval", type=float, default=0.3, help="状态保存间隔（秒）")
    
    args = parser.parse_args()
    
    # 更新数据共享目录
    DATA_DIR = args.data_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"数据共享目录: {DATA_DIR}")
    
    # 更新保存间隔
    shared_memory["save_interval"] = args.save_interval
    print(f"状态保存间隔: {shared_memory['save_interval']}秒")
    
    try:
        # 启动WebSocket服务器
        asyncio.run(run_websocket_server(args.host, args.port, args.model, args.use_onnx))
    except KeyboardInterrupt:
        print("\n用户中断，服务器关闭")
        # 打印最终网络统计
        print_network_stats()
    except Exception as e:
        print(f"服务器错误: {e}")

if __name__ == "__main__":
    main() 
