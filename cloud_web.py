#!/usr/bin/env python3
# Web服务器: 展示目标检测结果的Web界面

import os
import time
import json
import cv2
import numpy as np
import argparse
import threading
from flask import Flask, render_template, Response, jsonify, request
import pickle
from datetime import datetime

# Web服务器配置
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
PUBLIC_URL = "webzb.sh1.p2link.cn:8080"  # 公网访问地址

# 数据共享目录 - 用于与cloud_linux.py共享数据
DATA_DIR = "./shared_data"

# 创建Flask应用
app = Flask(__name__)

# 默认数据，当文件不可用或没有数据时使用
default_data = {
    "latest_detections": [],
    "latest_frame": None,
    "latest_stats": {
        "fps": 0,
        "total_frames": 0,
        "avg_process_time": 0,
        "last_detection_time": "-"
    },
    "network_stats": {
        "receive_times": [],
        "process_times": [],
        "send_times": [],
        "total_times": []
    },
    "last_update": time.time()
}

# 缓存数据和锁
cache = {
    "latest_detections": None,
    "latest_frame": None,
    "latest_stats": None,
    "network_stats": None,
    "last_read_time": {},  # 记录每种数据的最后读取时间
    "cache_valid_time": 0.2  # 缓存有效期（秒）
}
cache_lock = threading.Lock()

def get_data_from_file(filename, default=None):
    """从文件获取数据 - 简化版，带内存缓存"""
    cache_key = os.path.splitext(filename)[0]  # 去掉扩展名作为缓存键
    
    # 检查缓存是否有效
    with cache_lock:
        current_time = time.time()
        if (cache_key in cache and cache[cache_key] is not None and
            cache_key in cache["last_read_time"] and 
            current_time - cache["last_read_time"][cache_key] < cache["cache_valid_time"]):
            return cache[cache_key]
    
    # 缓存无效，从文件读取
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return default
    
    try:
        with open(filepath, 'rb') as f:
            try:
                data = pickle.load(f)
                
                # 更新缓存
                with cache_lock:
                    cache[cache_key] = data
                    cache["last_read_time"][cache_key] = time.time()
                
                return data
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"读取文件 {filename} 错误: {e}")
                return default
    except Exception as e:
        print(f"从文件获取数据失败: {e}")
        return default

def read_image_file(filename, default=None):
    """读取图像文件 - 简化版，带内存缓存"""
    cache_key = "latest_frame"
    
    # 检查缓存是否有效
    with cache_lock:
        current_time = time.time()
        if (cache[cache_key] is not None and 
            cache_key in cache["last_read_time"] and 
            current_time - cache["last_read_time"][cache_key] < cache["cache_valid_time"]):
            return cache[cache_key]
    
    # 缓存无效，从文件读取
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return default
    
    try:
        with open(filepath, 'rb') as f:
            img_data = f.read()
            
            # 解码图像
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # 更新缓存
                with cache_lock:
                    cache[cache_key] = img
                    cache["last_read_time"][cache_key] = time.time()
                
                return img
            return default
    except Exception as e:
        print(f"读取图像文件失败: {e}")
        return default

def get_latest_data():
    """获取最新的检测数据和统计信息"""
    # 从文件获取数据
    latest_detections = get_data_from_file("latest_detections.pkl", default_data["latest_detections"])
    latest_frame = read_image_file("latest_frame.jpg", None)
    latest_stats = get_data_from_file("latest_stats.pkl", default_data["latest_stats"])
    network_stats = get_data_from_file("network_stats.pkl", default_data["network_stats"])
    
    return {
        "latest_detections": latest_detections,
        "latest_frame": latest_frame,
        "latest_stats": latest_stats,
        "network_stats": network_stats,
        "last_update": time.time()
    }

# Flask路由 - Web界面
@app.route('/')
def index():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>云端目标检测系统</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {
                --bg: #eef3f8;
                --panel: #ffffff;
                --ink: #172033;
                --muted: #637083;
                --line: #dbe4f0;
                --brand: #2563eb;
                --brand-dark: #17345f;
                --cyan: #0891b2;
                --ok: #16a34a;
                --warn: #f59e0b;
                --accent: #7c3aed;
            }
            * {
                box-sizing: border-box;
            }
            body {
                font-family: "Microsoft YaHei", Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(180deg, #edf5ff 0, #f6f8fb 280px);
                color: var(--ink);
            }
            .container {
                width: min(1760px, calc(100vw - 32px));
                margin: 0 auto;
                padding: 18px 0;
            }
            .header {
                background: linear-gradient(135deg, var(--brand-dark), var(--brand) 58%, var(--cyan));
                color: white;
                padding: 28px 32px;
                border-radius: 8px;
                margin-bottom: 18px;
                box-shadow: 0 18px 45px rgba(31, 74, 140, 0.22);
            }
            .header-top {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 18px;
            }
            h1, h2, h3, p {
                margin-top: 0;
            }
            h1 {
                margin-bottom: 12px;
                font-size: 34px;
                letter-spacing: 0;
            }
            h2 {
                margin-bottom: 14px;
                font-size: 24px;
            }
            h3 {
                margin-bottom: 12px;
                font-size: 18px;
            }
            .subtitle {
                margin: 0;
                color: rgba(255, 255, 255, 0.88);
                font-size: 16px;
            }
            .members {
                margin: 0;
                padding: 8px 12px;
                border: 1px solid rgba(255, 255, 255, 0.32);
                border-radius: 6px;
                background: rgba(255, 255, 255, 0.12);
                color: white;
                white-space: nowrap;
            }
            .content {
                display: grid;
                grid-template-columns: minmax(560px, 1.45fr) minmax(300px, 0.75fr) minmax(340px, 0.85fr);
                gap: 16px;
                align-items: start;
            }
            .video-container, .stats-card, .detection-container {
                background-color: var(--panel);
                padding: 18px;
                border: 1px solid rgba(219, 228, 240, 0.9);
                border-radius: 8px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            }
            .video-header, .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 12px;
            }
            .video-header h2, .panel-header h2 {
                margin-bottom: 0;
            }
            .video-shell {
                background: #0f172a;
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }
            #video-feed {
                display: block;
                width: 100%;
                max-height: calc(100vh - 250px);
                object-fit: contain;
                border-radius: 0;
            }
            .stats-container {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .stats-grid, .network-stats-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }
            .stat-item {
                padding: 13px 10px;
                background: linear-gradient(180deg, #f8fafc, #f2f6fb);
                border: 1px solid #edf2f7;
                border-radius: 8px;
                text-align: center;
                min-height: 92px;
            }
            .stat-value {
                font-size: 25px;
                font-weight: 800;
                color: var(--brand);
                margin: 6px 0;
            }
            .stat-label {
                color: var(--muted);
                font-size: 13px;
            }
            .network-stats {
                margin-top: 18px;
                padding-top: 16px;
                border-top: 1px solid var(--line);
            }
            .detection-container {
                position: sticky;
                top: 12px;
            }
            .detection-list {
                max-height: calc(100vh - 210px);
                overflow-y: auto;
                padding-right: 4px;
            }
            .detection-item {
                padding: 12px;
                margin-bottom: 10px;
                background: #f8fafc;
                border: 1px solid #e6edf5;
                border-left: 4px solid var(--brand);
                border-radius: 7px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .detection-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 8px;
            }
            .detection-badges {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 6px;
            }
            .detection-distance, .detection-avg-distance {
                color: white;
                padding: 3px 8px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                white-space: nowrap;
            }
            .detection-distance {
                background-color: var(--warn);
            }
            .detection-avg-distance {
                background-color: var(--accent);
            }
            .refresh-button {
                background-color: var(--ok);
                color: white;
                border: none;
                padding: 10px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 700;
                transition: filter 0.2s, transform 0.2s;
            }
            .refresh-button:hover {
                filter: brightness(1.05);
            }
            .refresh-button:active {
                transform: translateY(1px);
            }
            .status {
                padding: 5px 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                display: inline-block;
            }
            .status-online {
                background-color: var(--ok);
                color: white;
            }
            .status-offline {
                background-color: #ef4444;
                color: white;
            }
            .footer {
                margin-top: 14px;
                text-align: center;
                color: var(--muted);
                font-size: 13px;
            }
            .loading {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(15, 23, 42, 0.42);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                color: white;
                font-size: 18px;
                visibility: hidden;
            }
            .loading.active {
                visibility: visible;
            }
            .spinner {
                border: 5px solid rgba(255, 255, 255, 0.45);
                border-top: 5px solid white;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @media (max-width: 1320px) {
                .content {
                    grid-template-columns: minmax(520px, 1fr) minmax(320px, 0.8fr);
                }
                .detection-container {
                    grid-column: 1 / -1;
                    position: static;
                }
                .detection-list {
                    max-height: 360px;
                }
            }
            @media (max-width: 900px) {
                .container {
                    width: calc(100vw - 20px);
                }
                .header-top {
                    align-items: flex-start;
                    flex-direction: column;
                }
                .members {
                    white-space: normal;
                }
                .content {
                    grid-template-columns: 1fr;
                }
                .stats-grid, .network-stats-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
    </head>
    <body>
        <div class="loading" id="loading-overlay">
            <div class="spinner"></div>
            <div>加载中...</div>
        </div>
        
        <div class="container">
            <div class="header">
                <div class="header-top">
                    <div>
                        <h1>云端目标检测系统</h1>
                        <p class="subtitle">实时检测统计和结果展示 - <span id="connection-status" class="status status-offline">检查连接中...</span></p>
                    </div>
                    <p class="members">小组成员：洪钰淇、丁宁、任筱舒</p>
                </div>
            </div>
            
            <div class="content">
                <div class="video-container">
                    <div class="video-header">
                        <h2>实时视频流</h2>
                        <button class="refresh-button" onclick="refreshVideoFeed(true)">刷新图像</button>
                    </div>
                    <div class="video-shell">
                        <img id="video-feed" src="/video_feed" alt="视频流">
                    </div>
                </div>
                
                <div class="stats-container">
                    <div class="stats-card">
                        <div class="panel-header">
                            <h2>系统状态</h2>
                        </div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-label">处理速度</div>
                                <div class="stat-value" id="fps">0</div>
                                <div class="stat-label">帧/秒</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">总处理帧数</div>
                                <div class="stat-value" id="total-frames">0</div>
                                <div class="stat-label">帧</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">平均处理时间</div>
                                <div class="stat-value" id="avg-process-time">0</div>
                                <div class="stat-label">毫秒</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">最近检测时间</div>
                                <div class="stat-value" id="last-detection-time">-</div>
                            </div>
                        </div>
                        
                        <div class="network-stats">
                            <h3>网络统计</h3>
                            <div class="network-stats-grid">
                                <div class="stat-item">
                                    <div class="stat-label">接收延迟</div>
                                    <div class="stat-value" id="receive-time">0</div>
                                    <div class="stat-label">毫秒</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-label">处理延迟</div>
                                    <div class="stat-value" id="process-time">0</div>
                                    <div class="stat-label">毫秒</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-label">发送延迟</div>
                                    <div class="stat-value" id="send-time">0</div>
                                    <div class="stat-label">毫秒</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-label">总延迟</div>
                                    <div class="stat-value" id="total-time">0</div>
                                    <div class="stat-label">毫秒</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="detection-container">
                    <div class="panel-header">
                        <h2>检测结果</h2>
                    </div>
                    <div id="detection-list" class="detection-list">
                        <div class="detection-item">等待检测结果...</div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>云端目标检测系统 | 数据更新时间: <span id="update-time">-</span></p>
            </div>
        </div>
        
        <script>
            let lastUpdateTime = 0;
            
            function showLoading() {
                document.getElementById('loading-overlay').classList.add('active');
            }
            
            function hideLoading() {
                document.getElementById('loading-overlay').classList.remove('active');
            }
            
            function refreshData(showLoadingIndicator = false) {
                if (showLoadingIndicator) {
                    showLoading();
                }
                
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        // 更新连接状态
                        const connectionStatus = document.getElementById('connection-status');
                        const timeSinceUpdate = (Date.now() - data.last_update * 1000) / 1000;
                        
                        if (timeSinceUpdate < 10) {
                            connectionStatus.textContent = '在线';
                            connectionStatus.className = 'status status-online';
                        } else {
                            connectionStatus.textContent = '离线';
                            connectionStatus.className = 'status status-offline';
                        }
                        
                        // 更新基本统计信息
                        document.getElementById('fps').textContent = data.fps.toFixed(1);
                        document.getElementById('total-frames').textContent = data.total_frames;
                        document.getElementById('avg-process-time').textContent = data.avg_process_time.toFixed(1);
                        document.getElementById('last-detection-time').textContent = data.last_detection_time;
                        
                        // 更新网络统计
                        document.getElementById('receive-time').textContent = data.receive_time.toFixed(1);
                        document.getElementById('process-time').textContent = data.process_time.toFixed(1);
                        document.getElementById('send-time').textContent = data.send_time.toFixed(1);
                        document.getElementById('total-time').textContent = data.total_time.toFixed(1);
                        
                        // 更新检测结果列表
                        const detectionList = document.getElementById('detection-list');
                        detectionList.innerHTML = '';
                        
                        if (data.detections.length === 0) {
                            detectionList.innerHTML = '<div class="detection-item">没有检测到物体</div>';
                        } else {
                            data.detections.forEach((det, index) => {
                                const item = document.createElement('div');
                                item.className = 'detection-item';
                                
                                // 添加检测结果标题和距离信息
                                const header = document.createElement('div');
                                header.className = 'detection-header';
                                
                                const title = document.createElement('strong');
                                title.textContent = `#${index+1} ${det.l} (${(det.c * 100).toFixed(1)}%)`;
                                
                                const distanceContainer = document.createElement('div');
                                distanceContainer.className = 'detection-badges';
                                
                                const distance = document.createElement('span');
                                distance.className = 'detection-distance';
                                distance.textContent = `当前: ${det.d}px`;
                                
                                const avgDistance = document.createElement('span');
                                avgDistance.className = 'detection-avg-distance';
                                avgDistance.textContent = `平均: ${det.a}px`;
                                
                                distanceContainer.appendChild(distance);
                                distanceContainer.appendChild(avgDistance);
                                
                                header.appendChild(title);
                                header.appendChild(distanceContainer);
                                
                                // 添加边界框信息
                                const boxInfo = document.createElement('div');
                                boxInfo.textContent = `边界框: [${det.b.join(', ')}]`;
                                
                                item.appendChild(header);
                                item.appendChild(boxInfo);
                                
                                detectionList.appendChild(item);
                            });
                        }
                        
                        // 更新数据更新时间
                        const now = new Date();
                        document.getElementById('update-time').textContent = now.toLocaleTimeString();
                        
                        lastUpdateTime = Date.now();
                        
                        if (showLoadingIndicator) {
                            hideLoading();
                        }
                    })
                    .catch(error => {
                        console.error('获取数据失败:', error);
                        const connectionStatus = document.getElementById('connection-status');
                        connectionStatus.textContent = '连接错误';
                        connectionStatus.className = 'status status-offline';
                        
                        if (showLoadingIndicator) {
                            hideLoading();
                        }
                    });
            }
            
            function refreshVideoFeed(showLoadingIndicator = false) {
                if (showLoadingIndicator) {
                    showLoading();
                }
                const videoFeed = document.getElementById('video-feed');
                if (videoFeed) {
                    // 添加时间戳参数避免浏览器缓存
                    videoFeed.src = `/video_feed?t=${Date.now()}`;
                    
                    // 当图像加载完成后隐藏加载指示器
                    videoFeed.onload = function() {
                        if (showLoadingIndicator) {
                            hideLoading();
                        }
                    };
                    
                    // 如果图像加载失败，也要隐藏加载指示器
                    videoFeed.onerror = function() {
                        if (showLoadingIndicator) {
                            hideLoading();
                        }
                    };
                }
            }
            
            // 页面加载时刷新一次
            document.addEventListener('DOMContentLoaded', function() {
                refreshData();
                
                // 设置定时器，每0.5秒自动刷新一次数据和画面
                setInterval(refreshData, 500);
                setInterval(function() {
                    refreshVideoFeed(false);
                }, 500);
            });
        </script>
    </body>
    </html>
    """

# 视频帧缓存
video_frame_cache = {
    "frame": None,
    "last_update": 0,
    "max_age": 0.1  # 最大缓存时间（秒）
}
video_cache_lock = threading.Lock()

@app.route('/video_feed')
def video_feed():
    """获取最新的视频帧"""
    # 检查是否可以使用缓存的视频帧
    current_time = time.time()
    with video_cache_lock:
        if (video_frame_cache["frame"] is not None and 
            current_time - video_frame_cache["last_update"] < video_frame_cache["max_age"]):
            return Response(video_frame_cache["frame"], mimetype='image/jpeg')
    
    # 获取新的视频帧
    data = get_latest_data()
    frame = data["latest_frame"]
    
    if frame is None:
        # 如果没有可用的帧，返回一个空白图像
        blank_image = np.ones((640, 640, 3), np.uint8) * 255
        cv2.putText(blank_image, "Waiting for stream...", (130, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        _, buffer = cv2.imencode('.jpg', blank_image)
        frame_bytes = buffer.tobytes()
    else:
        # 在图像上绘制检测结果
        detections = data["latest_detections"]
        if detections:
            frame_copy = frame.copy()
            for det in detections:
                label = det["l"]
                confidence = det["c"]
                box = det["b"]
                x1, y1, x2, y2 = box
                
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 添加FPS和处理时间信息
            stats = data["latest_stats"]
            cv2.putText(frame_copy, f"FPS: {stats['fps']:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_copy, f"Process: {stats['avg_process_time']:.1f}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 使用较低的JPEG质量以减少文件大小和传输时间
            _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
        else:
            # 使用较低的JPEG质量以减少文件大小和传输时间
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
    
    # 更新视频帧缓存
    with video_cache_lock:
        video_frame_cache["frame"] = frame_bytes
        video_frame_cache["last_update"] = current_time
    
    # 返回JPEG图像
    return Response(frame_bytes, mimetype='image/jpeg')

@app.route('/api/stats')
def get_stats():
    """获取当前统计数据和检测结果的API"""
    data = get_latest_data()
    
    # 计算网络统计平均值
    network_stats = data["network_stats"]
    receive_time = 0
    process_time = 0
    send_time = 0
    total_time = 0
    
    if network_stats["receive_times"]:
        receive_time = sum(network_stats["receive_times"]) / len(network_stats["receive_times"])
    if network_stats["process_times"]:
        process_time = sum(network_stats["process_times"]) / len(network_stats["process_times"])
    if network_stats["send_times"]:
        send_time = sum(network_stats["send_times"]) / len(network_stats["send_times"])
    if network_stats["total_times"]:
        total_time = sum(network_stats["total_times"]) / len(network_stats["total_times"])
    
    stats_data = {
        "fps": data["latest_stats"]["fps"],
        "total_frames": data["latest_stats"]["total_frames"],
        "avg_process_time": data["latest_stats"]["avg_process_time"],
        "last_detection_time": data["latest_stats"]["last_detection_time"],
        "detections": data["latest_detections"],
        "receive_time": receive_time,
        "process_time": process_time,
        "send_time": send_time,
        "total_time": total_time,
        "last_update": data["last_update"]
    }
    
    return jsonify(stats_data)

def main():
    """主函数"""
    # 声明全局变量
    global DATA_DIR, cache
    
    parser = argparse.ArgumentParser(description="Web界面: 展示目标检测结果")
    parser.add_argument("--host", type=str, default=WEB_HOST, help="Web服务器监听地址")
    parser.add_argument("--port", type=int, default=WEB_PORT, help="Web服务器监听端口，默认8080")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据共享目录路径")
    parser.add_argument("--cache-time", type=float, default=0.2, help="缓存有效时间（秒）")
    
    args = parser.parse_args()
    
    # 更新数据共享目录
    DATA_DIR = args.data_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"数据共享目录: {DATA_DIR}")
    
    # 更新缓存设置
    cache["cache_valid_time"] = args.cache_time
    video_frame_cache["max_age"] = args.cache_time / 2  # 视频帧缓存时间为普通缓存的一半
    print(f"缓存有效时间: {cache['cache_valid_time']}秒")
    
    # 启动Web服务器
    print(f"Web服务器启动，访问 http://{args.host}:{args.port} 或 http://{PUBLIC_URL}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main() 
