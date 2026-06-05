import argparse
import time

import numpy as np

from python_impl import create_gaussian_kernel, convolve5x5


def time_call(fn, repeat):
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn()
    return (time.perf_counter() - start) / repeat, result


def main():
    parser = argparse.ArgumentParser(description="Compare pure Python and C++ 5x5 convolution speed.")
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    image = rng.random((args.height, args.width), dtype=np.float32)
    kernel = create_gaussian_kernel(sigma=1.0).astype(np.float32)

    py_time, py_result = time_call(lambda: convolve5x5(image, kernel), args.repeat)
    print(f"Pure Python average: {py_time * 1000:.3f} ms")

    try:
        import filter_module
    except ImportError as exc:
        print(f"C++ module import failed: {exc}")
        print("Build it first with: python -m pip install -e .")
        return

    cpp_time, cpp_result = time_call(lambda: filter_module.convolve5x5(image, kernel), args.repeat)
    max_error = float(np.max(np.abs(np.asarray(cpp_result) - py_result)))
    speedup = py_time / cpp_time if cpp_time > 0 else float("inf")

    print(f"C++ extension average: {cpp_time * 1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max absolute error: {max_error:.8f}")


if __name__ == "__main__":
    main()
