import os
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

import param

def counter(func):
    is_first = True
    ax = None
    @functools.wraps(func)
    def counted(*args):
        nonlocal is_first, ax
        if is_first:
            is_first = False
            fig, ax = plt.subplots(1, 1)
        result = func(*args, ax=ax)
        return result
    return counted

@counter
def plot_anomalies(anomaly, *, ax):
    ax.cla()
    ax.set_ylim(-0.02, 1.02)
    ax.plot(anomaly)
    plt.pause(.0001)

def plot_features(waveform, features, filename, setting):
    fig, axes = plt.subplots(setting["enc"]["featureCount"]+1, 1)

    axes[0].cla()
    axes[0].plot(waveform)
    for i, feature in enumerate(features.T, start=1):
        axes[i].cla()
        axes[i].plot(feature)

def plot_input_data(inp):
    setting = param.default_parameters
    plt.imshow(inp.dense.reshape(setting["enc"]["featureCount"], -1))
    plt.pause(0.1)

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