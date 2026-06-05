import numpy as np


def create_gaussian_kernel(sigma=1.0):
    kernel = np.zeros((5, 5), dtype=np.float32)
    total = 0.0
    for i in range(-2, 3):
        for j in range(-2, 3):
            value = np.exp(-(i * i + j * j) / (2 * sigma * sigma))
            kernel[i + 2, j + 2] = value
            total += value
    return kernel / total


def create_sobel_kernels():
    kernel_x = np.array([
        [-1, -2, 0, 2, 1],
        [-4, -8, 0, 8, 4],
        [-6, -12, 0, 12, 6],
        [-4, -8, 0, 8, 4],
        [-1, -2, 0, 2, 1],
    ], dtype=np.float32) / 128.0

    kernel_y = np.array([
        [-1, -4, -6, -4, -1],
        [-2, -8, -12, -8, -2],
        [0, 0, 0, 0, 0],
        [2, 8, 12, 8, 2],
        [1, 4, 6, 4, 1],
    ], dtype=np.float32) / 128.0
    return kernel_x, kernel_y


def convolve5x5(image, kernel):
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            total = 0.0
            for ki in range(-2, 3):
                for kj in range(-2, 3):
                    ii = i + ki
                    jj = j + kj
                    if 0 <= ii < height and 0 <= jj < width:
                        total += image[ii, jj] * kernel[ki + 2, kj + 2]
            output[i, j] = total
    return output
