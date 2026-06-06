#!/usr/bin/env python3
"""WebSocket detection server using RKNNLite YOLO11 inference."""

import argparse
import asyncio
import base64
import ctypes
import json
import os
import pickle
import threading
import time

import cv2
import numpy as np
import websockets


DEFAULT_WS_PORT = 8765
DATA_DIR = "./shared_data"
IMG_SIZE = (640, 640)
OBJ_THRESH = 0.5
NMS_THRESH = 0.45

CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


RKNNLite = None
model = None
processed_frames = 0
last_report_time = time.time()

network_stats = {
    "receive_times": [],
    "process_times": [],
    "send_times": [],
    "total_times": [],
    "json_serialize_times": [],
    "send_start_times": [],
    "send_complete_times": [],
    "response_sizes": [],
}
stats_lock = threading.Lock()

latest_detections = []
latest_stats = {
    "fps": 0,
    "total_frames": 0,
    "avg_process_time": 0,
    "last_detection_time": "-",
}
web_data_lock = threading.Lock()

shared_memory = {
    "latest_frame_jpg": None,
    "last_save_time": 0,
    "save_interval": 0.3,
}

distance_history = {
    "values": [],
    "max_history": 20,
}


def ensure_rknnlite_loaded():
    global RKNNLite

    if RKNNLite is not None:
        return

    runtime_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rknn_runtime", "librknnrt.so")
    if os.path.exists(runtime_path):
        try:
            ctypes.CDLL(runtime_path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            print(f"已预加载项目内RKNN runtime: {runtime_path}")
        except OSError as exc:
            print(f"预加载项目内RKNN runtime失败，将使用系统runtime: {exc}")

    from rknnlite.api import RKNNLite as RKNNLiteClass
    RKNNLite = RKNNLiteClass


def save_data_to_file(filename, data):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as exc:
        print(f"保存数据到文件失败: {exc}")
        return False


def save_image_bytes(filename, img_bytes):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception as exc:
        print(f"保存图像到文件失败: {exc}")
        return False


def print_network_stats():
    with stats_lock:
        if not network_stats["total_times"]:
            print("尚无网络统计数据")
            return

        avg_receive = sum(network_stats["receive_times"]) / len(network_stats["receive_times"])
        avg_process = sum(network_stats["process_times"]) / len(network_stats["process_times"])
        avg_send = sum(network_stats["send_times"]) / len(network_stats["send_times"])
        avg_total = sum(network_stats["total_times"]) / len(network_stats["total_times"])
        avg_json = (
            sum(network_stats["json_serialize_times"]) / len(network_stats["json_serialize_times"])
            if network_stats["json_serialize_times"]
            else 0
        )
        avg_size = (
            sum(network_stats["response_sizes"]) / len(network_stats["response_sizes"])
            if network_stats["response_sizes"]
            else 0
        )

        print("\n===== RKNN服务器端延迟统计 =====")
        print(f"接收请求延迟 (平均): {avg_receive:.1f}ms")
        print(f"处理时间 (平均): {avg_process:.1f}ms")
        print(f"JSON序列化时间 (平均): {avg_json:.1f}ms")
        print(f"总发送时间 (平均): {avg_send:.1f}ms")
        print(f"总处理时间 (平均): {avg_total:.1f}ms")
        print(f"响应大小 (平均): {avg_size:.1f}字节 ({avg_size / 1024:.2f}KB)")
        print("==============================\n")

        with web_data_lock:
            latest_stats["avg_process_time"] = avg_process


def save_state_periodically():
    while True:
        try:
            current_time = time.time()
            if current_time - shared_memory["last_save_time"] >= shared_memory["save_interval"]:
                save_data_to_file("latest_detections.pkl", latest_detections)
                save_data_to_file("latest_stats.pkl", latest_stats)
                save_data_to_file("network_stats.pkl", network_stats)
                if shared_memory["latest_frame_jpg"] is not None:
                    save_image_bytes("latest_frame.jpg", shared_memory["latest_frame_jpg"])
                shared_memory["last_save_time"] = current_time
                print(f"状态已保存到文件，时间: {time.strftime('%H:%M:%S')}")

            time.sleep(max(0.05, min(shared_memory["save_interval"], 1.0)))
        except Exception as exc:
            print(f"保存状态失败: {exc}")
            time.sleep(5)


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    keep = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[keep]
    return boxes[keep], classes[keep], scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1 + 0.00001)
        inter_h = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = inter_w * inter_h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(overlap <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


def flatten_output(value):
    channels = value.shape[1]
    value = value.transpose(0, 2, 3, 1)
    return value.reshape(-1, channels)


def post_process(outputs):
    boxes = []
    scores = []
    classes_conf = []
    branch_count = 3
    pair_per_branch = len(outputs) // branch_count

    for i in range(branch_count):
        boxes.append(box_process(outputs[pair_per_branch * i]))
        classes_conf.append(outputs[pair_per_branch * i + 1])
        scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    boxes = np.concatenate([flatten_output(v) for v in boxes])
    classes_conf = np.concatenate([flatten_output(v) for v in classes_conf])
    scores = np.concatenate([flatten_output(v) for v in scores])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    if boxes.size == 0:
        return None, None, None

    nboxes = []
    nclasses = []
    nscores = []
    for class_id in set(classes):
        inds = np.where(classes == class_id)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    order = scores.argsort()[::-1]
    return boxes[order], classes[order], scores[order]


def outputs_to_detections(outputs, image_shape):
    boxes, classes, scores = post_process(outputs)
    if boxes is None:
        return []

    detections = []
    image_center_x = image_shape[1] / 2
    image_center_y = image_shape[0] / 2

    for box, class_id, confidence in zip(boxes, classes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(image_shape[1] - 1, x1))
        y1 = max(0, min(image_shape[0] - 1, y1))
        x2 = max(0, min(image_shape[1] - 1, x2))
        y2 = max(0, min(image_shape[0] - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        distance_to_center = float(np.sqrt((box_center_x - image_center_x) ** 2 + (box_center_y - image_center_y) ** 2))
        distance_history["values"].append(distance_to_center)
        if len(distance_history["values"]) > distance_history["max_history"]:
            distance_history["values"] = distance_history["values"][-distance_history["max_history"]:]
        avg_distance = sum(distance_history["values"]) / len(distance_history["values"])

        label = CLASSES[int(class_id)] if int(class_id) < len(CLASSES) else str(int(class_id))
        detections.append({
            "l": label,
            "c": round(float(confidence), 3),
            "b": [x1, y1, x2, y2],
            "d": round(distance_to_center, 1),
            "a": round(avg_distance, 1),
        })

    return detections[:50]


def encode_latest_frame(img):
    ok, img_encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        shared_memory["latest_frame_jpg"] = img_encoded.tobytes()


async def process_image(websocket, path=None):
    global processed_frames, last_report_time, latest_detections, latest_stats

    client_address = getattr(websocket, "remote_address", "unknown")
    print(f"客户端已连接: {client_address}")
    frame_count = 0
    stats_timer = time.time()

    try:
        async for message in websocket:
            receive_start = time.time()
            try:
                data = json.loads(message)
                img_base64 = data.get("image")
                receive_end = time.time()
                receive_time = (receive_end - receive_start) * 1000

                if not img_base64:
                    await websocket.send(json.dumps({"error": "未收到图像数据"}))
                    continue

                img_bytes = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    await websocket.send(json.dumps({"error": "图像解码失败"}))
                    continue

                img = cv2.resize(img, IMG_SIZE)
                with web_data_lock:
                    encode_latest_frame(img)

                frame_count += 1
                processed_frames += 1
                current_time = time.time()
                if current_time - last_report_time >= 1.0:
                    fps = processed_frames / (current_time - last_report_time)
                    print(f"RKNN处理速度: {fps:.1f} FPS (当前连接 {frame_count} 帧)")
                    processed_frames = 0
                    last_report_time = current_time
                    with web_data_lock:
                        latest_stats["fps"] = fps
                        latest_stats["total_frames"] = frame_count

                if current_time - stats_timer >= 10.0:
                    print_network_stats()
                    stats_timer = current_time

                input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(input_data, axis=0)
                inference_start = time.time()
                outputs = model.inference(inputs=[input_data])
                inference_time = time.time() - inference_start

                detections = outputs_to_detections(outputs, img.shape)
                with web_data_lock:
                    latest_detections = detections
                    latest_stats["last_detection_time"] = time.strftime("%H:%M:%S", time.localtime())

                process_time = (time.time() - receive_end) * 1000
                result = {
                    "f": frame_count,
                    "n": len(detections),
                    "d": detections,
                    "i": round(inference_time, 4),
                    "t": {
                        "r": round(receive_time, 1),
                        "p": round(process_time, 1),
                        "t": round(receive_time + process_time, 1),
                    },
                }

                json_start = time.time()
                json_result = json.dumps(result)
                response_size = len(json_result.encode("utf-8"))
                json_time = (time.time() - json_start) * 1000
                send_start = time.time()
                await websocket.send(json_result)
                send_time = (time.time() - send_start) * 1000
                total_time = receive_time + process_time + send_time

                with stats_lock:
                    network_stats["receive_times"].append(receive_time)
                    network_stats["process_times"].append(process_time)
                    network_stats["send_times"].append(send_time)
                    network_stats["total_times"].append(total_time)
                    network_stats["json_serialize_times"].append(json_time)
                    network_stats["send_start_times"].append(0)
                    network_stats["send_complete_times"].append(send_time)
                    network_stats["response_sizes"].append(response_size)
                    for key in network_stats:
                        if len(network_stats[key]) > 50:
                            network_stats[key] = network_stats[key][-50:]

                if detections:
                    print(f"第 {frame_count} 帧检测到 {len(detections)} 个对象")
                    print(f"接收: {receive_time:.1f}ms, RKNN处理: {process_time:.1f}ms, 发送: {send_time:.1f}ms")

            except json.JSONDecodeError:
                print("JSON解析错误")
                await websocket.send(json.dumps({"error": "JSON解析错误"}))
            except Exception as exc:
                print(f"处理图像时出错: {exc}")
                await websocket.send(json.dumps({"error": str(exc)}))

    except websockets.exceptions.ConnectionClosed:
        print(f"客户端断开连接: {client_address}")
        print_network_stats()
    except Exception as exc:
        print(f"WebSocket错误: {exc}")


def parse_core_mask(value):
    ensure_rknnlite_loaded()
    value = str(value).lower()
    if value in {"all", "0_1_2"}:
        return RKNNLite.NPU_CORE_0_1_2
    if value == "0":
        return RKNNLite.NPU_CORE_0
    if value == "1":
        return RKNNLite.NPU_CORE_1
    if value == "2":
        return RKNNLite.NPU_CORE_2
    return RKNNLite.NPU_CORE_0_1_2


def load_model(model_path, core_mask):
    global model

    ensure_rknnlite_loaded()

    if not os.path.exists(model_path):
        print(f"错误: RKNN模型文件不存在: {model_path}")
        return False

    model = RKNNLite()
    print(f"正在加载RKNN模型: {model_path}")
    ret = model.load_rknn(model_path)
    if ret != 0:
        print(f"RKNN模型加载失败: {ret}")
        return False

    print("正在初始化RKNN runtime")
    ret = model.init_runtime(core_mask=parse_core_mask(core_mask))
    if ret != 0:
        print(f"RKNN runtime初始化失败: {ret}")
        print("如果提示 Invalid RKNN model version，请更新板子上的 librknnrt.so。")
        return False

    dummy = np.zeros((1, IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    for _ in range(3):
        _ = model.inference(inputs=[dummy])
    print("RKNN模型加载和预热完成")
    return True


async def run_websocket_server(host, port, model_path, core_mask):
    if not load_model(model_path, core_mask):
        print("模型加载失败，服务器无法启动")
        return

    print(f"RKNN WebSocket服务器启动，监听 {host}:{port}")
    save_thread = threading.Thread(target=save_state_periodically, daemon=True)
    save_thread.start()

    async with websockets.serve(
        process_image,
        host,
        port,
        max_size=10 * 1024 * 1024,
        max_queue=32,
        compression=None,
        ping_interval=None,
        ping_timeout=None,
    ):
        print(f"WebSocket URL: ws://{host}:{port}")
        print("等待客户端连接...")
        await asyncio.Future()


def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="RKNN WebSocket目标检测服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT)
    parser.add_argument("--model", type=str, default="yolo11n_fp.rknn", help="RKNN model path")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--save-interval", type=float, default=0.3)
    parser.add_argument("--core-mask", type=str, default="all", help="all, 0, 1, or 2")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    shared_memory["save_interval"] = args.save_interval
    print(f"数据共享目录: {DATA_DIR}")
    print(f"状态保存间隔: {shared_memory['save_interval']}秒")

    try:
        asyncio.run(run_websocket_server(args.host, args.port, args.model, args.core_mask))
    except KeyboardInterrupt:
        print("\n用户中断，服务器关闭")
        print_network_stats()
    except Exception as exc:
        print(f"服务器错误: {exc}")
    finally:
        if model is not None:
            model.release()


if __name__ == "__main__":
    main()
