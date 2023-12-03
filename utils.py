import pandas as pd

from settings import SCREEN_SIZE


def normalize(
    x_in: any,
    y_in: any,
    x_out: any=None,
    y_out: any=None
    ) -> tuple:
        """
        Normalize the original positions between [0, 1] and the output values
        between [-1, 1].
        """
        x_in = x_in / SCREEN_SIZE[0]
        y_in = y_in / SCREEN_SIZE[1]

        if x_out is None and y_out is None:
            return x_in, y_in

        x_out = x_out / SCREEN_SIZE[0]
        y_out = y_out / SCREEN_SIZE[1]
        return x_in, y_in, x_out, y_out