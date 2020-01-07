import os
import itertools
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

import param

def plot_anomalies(anomaly):
    fig, ax = plt.subplots(1, 1)
    ax.plot(anomaly)
    plt.pause(0.001)
    plt.close()


def plot_features(waveform, features, filename, setting):
    fig, axes = plt.subplots(setting["enc"]["featureCount"]+1, 1)

    axes[0].cla()
    axes[0].plot(waveform)
    for i, feature in enumerate(features.T, start=1):
        axes[i].cla()
        axes[i].plot(feature)

    filename = os.path.basename(filename).replace(".wav", "") + ".png"
    plt.savefig(filename)

def write_gif_file(features, features_iter, filename):
    fig = plt.figure()
    axes, plots = [], []
    num_axes = len(features) - 1
    for i in range(num_axes):
        data = np.zeros_like(features_iter[0][i + 1])
        tmp_ax = fig.add_subplot(num_axes, 1, i + 1)
        tmp_ax.set_ylim(features[i + 1].min(), features[i + 1].max())
        plots.append(tmp_ax.plot(data)[0])
        axes.append(tmp_ax)

    def update(feature):
        for i, feat in enumerate(feature):
            if i == 0:
                axes[i].set_title("f0: " + str(feat))
            else:
                plots[i - 1].set_data(np.arange(feat.shape[0]), feat)

    ani = anime.FuncAnimation(fig, update, features_iter, interval=80, repeat_delay=1000)
    filename = os.path.basename(filename).replace('.wav', '')
    gif_name = filename + '.gif'
    ani.save(gif_name, writer="imagemagick")
    print("wrote to wav file :", gif_name)

def counter(func):
    is_first = True
    axes = None
    @functools.wraps(func)
    def counted(*args):
        nonlocal is_first, axes
        if is_first:
            is_first = False
            fig, axes = plt.subplots(1, 1+4)
        result = func(*args, axes=axes)
        return result
    return counted

@counter
def visualizeSDR(title, encoding, sdrs=None, *, axes):
    title = title[40:] if len(title) > 40 else title
    fmt = "{} columns:region{}"
    col_names = ["active", "predictive"]
    titles = [fmt.format(col_names[i%2], i//2) for i in range(len(sdrs))]
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
        _ = ax.set_aspect(10.0) if dim==1 else ax.set_aspect('equal')
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