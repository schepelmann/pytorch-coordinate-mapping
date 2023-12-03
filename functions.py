import numpy as np


def increase(x: np.ndarray, screen_size: np.ndarray) -> np.ndarray:
    x = x - screen_size / 2
    x *= 1.5
    x += screen_size // 2
    x = np.clip(x, 0, screen_size)

    return x


def decrease(x: np.ndarray, screen_size: np.ndarray) -> np.ndarray:
    x = x - screen_size / 2
    x *= 0.8
    x += screen_size // 2
    x = np.clip(x, 0, screen_size)

    return x


def reverse(x: np.ndarray, screen_size: np.ndarray) -> np.ndarray:
    x = x - screen_size / 2
    x = -x
    x += screen_size // 2
    x = np.clip(x, 0, screen_size)

    return x