# 5x5卷积 whl 加分模块

这个目录保留一个小型图像处理功能块：5x5卷积。它同时提供纯Python实现和C++/pybind11扩展实现，用于满足大作业加分项中“把一个小功能块用C/C++实现，编译成whl调用，并与Python实现比较性能”的要求。

## 文件说明

- `fliter.cpp`：C++ 5x5卷积、Gaussian核、Sobel核实现。
- `setup.py`：pybind11扩展构建脚本。
- `python_impl.py`：纯Python/Numpy对照实现。
- `benchmark_filter.py`：性能对比脚本，不依赖外部图片。

## 在OrangePi上构建

```bash
cd ~/code0604/my_wheel
python -m pip install pybind11 wheel numpy
python setup.py bdist_wheel
python -m pip install dist/image_filter-*.whl --force-reinstall
```

也可以开发模式安装：

```bash
python -m pip install -e .
```

## 性能对比

```bash
python benchmark_filter.py --height 120 --width 160 --repeat 3
```

报告里可以记录输出的 Pure Python average、C++ extension average、Speedup 和 Max absolute error。这个模块不作为YOLO主流程的必要依赖，只作为图像预处理/性能优化加分项展示。
