#!/usr/bin/env python3
"""WebSocket client used by main_cloud.py."""

import base64
import json
import os
import time

import cv2
import numpy as np
import websocket


DEFAULT_WS_URL = os.getenv("CLOUD_WS_URL", "ws://127.0.0.1:8765")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class CloudClient:
    def __init__(self, ws_url=DEFAULT_WS_URL, timeout=8, jpeg_quality=70):
        self.ws_url = ws_url
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self.ws = None
        self.CLASSES = COCO_CLASSES.copy()
        self.latest_result = None
        self.latest_boxes = None
        self.latest_classes = None
        self.latest_scores = None
        self.sent_frames = 0
        self.last_fps_time = time.time()
        self.connect()

    def connect(self):
        self.close(silent=True)
        print(f"尝试连接到WebSocket服务: {self.ws_url}")
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=self.timeout)
            print(f"WebSocket已连接到: {self.ws_url}")
            return True
        except Exception as exc:
            print(f"WebSocket连接失败: {exc}")
            self.ws = None
            return False

    def close(self, silent=False):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = None
        if not silent:
            print("WebSocket连接已关闭")

    def _ensure_connection(self):
        if self.ws is None:
            return self.connect()
        return True

    def _class_index(self, label):
        if label not in self.CLASSES:
            self.CLASSES.append(label)
            print(f"添加新类别: {label}, index={len(self.CLASSES) - 1}")
        return self.CLASSES.index(label)

    def _parse_response(self, message):
        result_json = json.loads(message)
        detections = result_json.get("d", [])

        boxes = []
        classes = []
        scores = []
        for detection in detections:
            label = detection.get("l")
            confidence = detection.get("c")
            bbox = detection.get("b")
            if not label or confidence is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [float(coord) for coord in bbox]
            if x1 >= x2 or y1 >= y2:
                continue

            boxes.append([x1, y1, x2, y2])
            classes.append(self._class_index(str(label)))
            scores.append(float(confidence))

        self.latest_result = result_json
        self.latest_boxes = np.array(boxes, dtype=np.float32) if boxes else None
        self.latest_classes = np.array(classes, dtype=np.int32) if classes else None
        self.latest_scores = np.array(scores, dtype=np.float32) if scores else None

    def send_image(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("输入图像为空，跳过发送")
            return False

        if frame.shape[0] != 640 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 640))

        ok, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            print("图像JPEG编码失败")
            return False

        img_bytes = img_encoded.tobytes()
        message = json.dumps({"image": base64.b64encode(img_bytes).decode("utf-8")})

        try:
            if not self._ensure_connection():
                return False
            send_start = time.time()
            self.ws.send(message)
            response = self.ws.recv()
            self._parse_response(response)

            self.sent_frames += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                fps = self.sent_frames / (now - self.last_fps_time)
                round_trip_ms = (now - send_start) * 1000
                print(f"发送速度: {fps:.1f} FPS, 图像大小: {len(img_bytes) / 1024:.1f} KB, 往返: {round_trip_ms:.1f} ms")
                self.sent_frames = 0
                self.last_fps_time = now
            return True
        except Exception as exc:
            print(f"发送图像数据失败: {exc}")
            self.close(silent=True)
            return False

    def get_latest_results(self):
        result = self.latest_result.copy() if isinstance(self.latest_result, dict) else self.latest_result
        boxes = self.latest_boxes.copy() if self.latest_boxes is not None else None
        classes = self.latest_classes.copy() if self.latest_classes is not None else None
        scores = self.latest_scores.copy() if self.latest_scores is not None else None
        return result, boxes, classes, scores
