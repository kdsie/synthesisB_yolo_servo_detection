#!/usr/bin/env python3
"""OrangePi camera sender with optional servo tracking.

Run this on the OrangePi after cloud_linux.py and cloud_web.py are running.
The cloud server performs YOLO inference; this script only captures frames,
selects one target box, and optionally drives the PCA9685 servo gimbal.
"""

from multiprocessing import Lock, Manager, Process
import argparse
import os
import time

import cv2

from cloud_client import CloudClient
from servo import servo_control


CHANNEL_X = 0
CHANNEL_Y = 15
FRAME_SIZE = 640
DEFAULT_WS_URL = os.getenv("CLOUD_WS_URL", "ws://127.0.0.1:8765")


def class_name(client, class_id):
    if 0 <= class_id < len(client.CLASSES):
        return client.CLASSES[class_id]
    return f"class_{class_id}"


def select_target_box(client, boxes, classes, scores, target_label=""):
    """Pick the box used by the servo.

    If target_label is set, only that class is tracked. Otherwise the highest
    confidence detection returned by the cloud server is used.
    """
    if boxes is None or len(boxes) == 0:
        return None, None, 0.0

    best_index = None
    best_score = -1.0
    target_label = (target_label or "").strip().lower()

    for idx, box in enumerate(boxes):
        class_id = int(classes[idx]) if classes is not None and idx < len(classes) else -1
        label = class_name(client, class_id)
        score = float(scores[idx]) if scores is not None and idx < len(scores) else 0.0

        if target_label and label.lower() != target_label:
            continue
        if score > best_score:
            best_index = idx
            best_score = score

    if best_index is None:
        return None, None, 0.0

    class_id = int(classes[best_index]) if classes is not None and best_index < len(classes) else -1
    label = class_name(client, class_id)
    return boxes[best_index], label, best_score


def camera_process(shared, lock, camera_id, ws_url, target_label, send_interval):
    print("进入camera_process")
    client = CloudClient(ws_url)
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头分辨率: {width}x{height}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("未获取到摄像头画面，等待重试...")
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            if not client.send_image(frame):
                time.sleep(1)
                continue

            result, boxes, classes, scores = client.get_latest_results()
            target_box, label, score = select_target_box(client, boxes, classes, scores, target_label)

            if target_box is not None:
                print(f"跟踪目标: {label} conf={score:.2f} box={target_box}")
            elif target_label:
                print(f"未检测到目标: {target_label}")
            else:
                print("未检测到可跟踪目标")

            with lock:
                shared["box"] = target_box
                shared["frame"] = frame
                shared["boxes"] = boxes
                shared["classes"] = classes
                shared["scores"] = scores

            time.sleep(max(0.0, send_interval))

    except KeyboardInterrupt:
        print("camera_process已中断")
    except Exception as exc:
        print(f"camera_process出错: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        client.close()


def servo_process(shared, lock):
    print("进入servo_process")
    time.sleep(3)

    try:
        import Adafruit_PCA9685
        servo_pwm = Adafruit_PCA9685.PCA9685(busnum=1)
        servo_pwm.set_pwm_freq(60)
        print("PCA9685初始化成功")

        angle_cur_x = 500
        angle_max_x = 700
        angle_min_x = 150
        angle_cur_y = 350
        angle_max_y = 500
        angle_min_y = 200

        servo_pwm.set_pwm(CHANNEL_X, 0, angle_cur_x)
        servo_pwm.set_pwm(CHANNEL_Y, 0, angle_cur_y)
        print(f"舵机设置到初始位置: X={angle_cur_x}, Y={angle_cur_y}")
    except Exception as exc:
        print(f"舵机初始化失败，跳过云台控制: {exc}")
        import traceback
        traceback.print_exc()
        return

    error_max = 25
    frame_center = (FRAME_SIZE // 2, FRAME_SIZE // 2)

    try:
        while True:
            with lock:
                box = shared.get("box", None)

            if box is None:
                time.sleep(0.05)
                continue

            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            delta_x = x_center - frame_center[0]
            delta_y = frame_center[1] - y_center

            if abs(delta_x) > error_max:
                angle_cur_x = servo_control(servo_pwm, CHANNEL_X, angle_cur_x, angle_max_x, angle_min_x, delta_x)
            if abs(delta_y) > error_max:
                angle_cur_y = servo_control(servo_pwm, CHANNEL_Y, angle_cur_y, angle_max_y, angle_min_y, delta_y)

            with lock:
                shared["box"] = None
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("servo_process已中断")
    except Exception as exc:
        print(f"servo_process出错: {exc}")
        import traceback
        traceback.print_exc()


def display_process(shared, lock):
    print("进入display_process")
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", FRAME_SIZE, FRAME_SIZE)

    try:
        while True:
            with lock:
                frame = shared.get("frame", None)
                boxes = shared.get("boxes", None)
                scores = shared.get("scores", None)

            if frame is None:
                time.sleep(0.05)
                continue

            display_frame = frame.copy()
            if boxes is not None:
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    score = float(scores[idx]) if scores is not None and idx < len(scores) else 0.0
                    color = (0, 0, 255) if idx == 0 else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{score:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            center = FRAME_SIZE // 2
            cv2.line(display_frame, (center - 20, center), (center + 20, center), (0, 0, 255), 2)
            cv2.line(display_frame, (center, center - 20), (center, center + 20), (0, 0, 255), 2)
            cv2.imshow("Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="OrangePi云端目标检测客户端")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="cloud_linux.py的WebSocket地址")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--target-label", default=os.getenv("TARGET_LABEL", ""), help="只跟踪指定类别；留空则跟踪最高置信度目标")
    parser.add_argument("--send-interval", type=float, default=0.05, help="两帧发送之间的额外等待时间")
    parser.add_argument("--no-servo", action="store_true", help="不启动舵机云台进程")
    parser.add_argument("--display", action="store_true", help="打开本地OpenCV显示窗口；SSH下通常不用")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"WebSocket服务: {args.ws_url}")
    if args.target_label:
        print(f"跟踪类别: {args.target_label}")
    else:
        print("跟踪类别: 最高置信度目标")

    manager = Manager()
    shared = manager.dict()
    shared["box"] = None
    shared["frame"] = None
    shared["boxes"] = None
    shared["classes"] = None
    shared["scores"] = None
    lock = Lock()

    processes = [Process(target=camera_process, args=(shared, lock, args.camera, args.ws_url, args.target_label, args.send_interval))]
    if not args.no_servo:
        processes.append(Process(target=servo_process, args=(shared, lock)))
    if args.display:
        processes.append(Process(target=display_process, args=(shared, lock)))

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("主进程已中断")
        for process in processes:
            if process.is_alive():
                process.terminate()
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()
