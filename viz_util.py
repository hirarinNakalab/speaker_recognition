import itertools
import functools

import numpy as np
import matplotlib.pyplot as plt

import param

def counter(func):
    is_first = True
    axes = None
    @functools.wraps(func)
    def counted(*args):
        nonlocal is_first, axes
        if is_first:
            is_first = False
            fig, axes = plt.subplots(3+1, 1)
        result = func(*args, axes=axes)
        return result
    return counted

@counter
def visualizeSDR(title, encoding, sdrs=None, *, axes):
    title = title[:60] if len(title) > 60 else title
    titles = ["active columns:region{}".format(i) for i in range(len(sdrs))]
    titles.insert(0, title)
    sdrs.insert(0, encoding)
    for ax, sdr, tit in zip(axes, sdrs, titles):
        ax.cla()
        ax.set_title(tit, size=12)
        if not sdr:
            continue
        dense = sdr.dense
        dim = dense.ndim
        nrow = sdr.dimensions[0] if dim >= 2 else 1
        ncol = sdr.dimensions[1] if dim >= 2 else sdr.dimensions[0]
        ax.set_xlim(-0.75, ncol + 0.75)
        ax.set_ylim(-0.75, nrow + 0.75)
        _ = ax.set_aspect(10.0) if param.dimension==1 else ax.set_aspect('equal')
        x, y = [], []
        if not dim == 2:
            dense = dense[np.newaxis, :]
        for row, col in itertools.product(range(nrow), range(ncol)):
            if dense[row, col] == 1:
                x.append(col);y.append(row)
        ax.scatter(x, y)
    plt.pause(0.1)


def visualize(i, title, encoding, outputs):
    if param.args["VIZ_COLS"]:
        visualizeSDR(title, encoding, outputs)
        if (i + 1) % 10 == 0:
            plt.savefig(param.output_img.format(i + 1))