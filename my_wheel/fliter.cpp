#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>

namespace py = pybind11;

// 5x5 卷积函数
py::array_t<float> convolve5x5(py::array_t<float> input, py::array_t<float> kernel) {
    // 获取输入数组信息
    py::buffer_info input_buf = input.request();
    py::buffer_info kernel_buf = kernel.request();
    
    // 检查内核尺寸是否为5x5
    if (kernel_buf.shape[0] != 5 || kernel_buf.shape[1] != 5) {
        throw std::runtime_error("Kernel shape must be 5x5");
    }
    
    // 检查输入维度
    if (input_buf.ndim != 2) {
        throw std::runtime_error("Input must be 2-dimensional");
    }
    
    // 获取输入图像尺寸
    size_t height = input_buf.shape[0];
    size_t width = input_buf.shape[1];
    
    // 创建输出数组
    py::array_t<float> output = py::array_t<float>({height, width});
    py::buffer_info output_buf = output.request();
    
    // 获取数据指针
    float* input_ptr = static_cast<float*>(input_buf.ptr);
    float* kernel_ptr = static_cast<float*>(kernel_buf.ptr);
    float* output_ptr = static_cast<float*>(output_buf.ptr);
    
    // 执行卷积
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            float sum = 0.0f;
            
            // 应用5x5卷积核
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    // 计算输入图像位置
                    int ii = i + ki;
                    int jj = j + kj;
                    
                    // 边界检查 - 使用零填充
                    if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
                        sum += input_ptr[ii * width + jj] * kernel_ptr[(ki + 2) * 5 + (kj + 2)];
                    }
                }
            }
            
            // 存储结果
            output_ptr[i * width + j] = sum;
        }
    }
    
    return output;
}

// 创建高斯核函数
py::array_t<float> create_gaussian_kernel(float sigma = 1.0) {
    // 创建5x5高斯核
    py::array_t<float> kernel({5, 5});
    py::buffer_info kernel_buf = kernel.request();
    float* kernel_ptr = static_cast<float*>(kernel_buf.ptr);
    
    // 计算高斯核
    float sum = 0.0f;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            float value = exp(-(i*i + j*j) / (2 * sigma * sigma));
            kernel_ptr[(i + 2) * 5 + (j + 2)] = value;
            sum += value;
        }
    }
    
    // 归一化
    for (int i = 0; i < 25; i++) {
        kernel_ptr[i] /= sum;
    }
    
    return kernel;
}

// 创建Sobel边缘检测核函数
std::pair<py::array_t<float>, py::array_t<float>> create_sobel_kernels() {
    // 创建Sobel X方向核
    py::array_t<float> kernel_x({5, 5});
    py::buffer_info kernel_x_buf = kernel_x.request();
    float* kernel_x_ptr = static_cast<float*>(kernel_x_buf.ptr);
    
    // 创建Sobel Y方向核
    py::array_t<float> kernel_y({5, 5});
    py::buffer_info kernel_y_buf = kernel_y.request();
    float* kernel_y_ptr = static_cast<float*>(kernel_y_buf.ptr);
    
    // 5x5 Sobel核定义
    float sobel_x[25] = {
        -1, -2, 0, 2, 1,
        -4, -8, 0, 8, 4,
        -6, -12, 0, 12, 6,
        -4, -8, 0, 8, 4,
        -1, -2, 0, 2, 1
    };
    
    float sobel_y[25] = {
        -1, -4, -6, -4, -1,
        -2, -8, -12, -8, -2,
        0, 0, 0, 0, 0,
        2, 8, 12, 8, 2,
        1, 4, 6, 4, 1
    };
    
    // 复制数据
    for (int i = 0; i < 25; i++) {
        kernel_x_ptr[i] = sobel_x[i] / 128.0f;
        kernel_y_ptr[i] = sobel_y[i] / 128.0f;
    }
    
    return std::make_pair(kernel_x, kernel_y);
}

// Python模块定义
PYBIND11_MODULE(filter_module, m) {
    m.doc() = "Image filtering module with 5x5 convolution";
    
    m.def("convolve5x5", &convolve5x5, 
          "Apply 5x5 convolution on an image",
          py::arg("input"), py::arg("kernel"));
    
    m.def("create_gaussian_kernel", &create_gaussian_kernel,
          "Create a 5x5 Gaussian kernel",
          py::arg("sigma") = 1.0);
    
    m.def("create_sobel_kernels", &create_sobel_kernels,
          "Create 5x5 Sobel kernels for edge detection");
}
