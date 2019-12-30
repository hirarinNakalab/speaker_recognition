import os
import itertools

import pyworld as pw
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

from layers import Region, Layer, createEncoder, createClassifier
from sdr_util import get_dense_array
import viz_util
import param


def get_wavfile_list(path):
    wav_files = []
    for dirpath, subdirs, files in os.walk(path):
        for x in files:
            if x.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, x))
    return wav_files

def update(feature):
    for i, feat in enumerate(feature):
        if i == 0:
            axes[i].set_title("f0: " + str(feat))
        else:
            plots[i-1].set_data(np.arange(feat.shape[0]), feat)


if __name__ == '__main__':
    data_path = param.input_file
    wav_files = get_wavfile_list(data_path)
    n_files = len(wav_files)
    split_idx = int(n_files*0.6)
    dataset = {'train': wav_files[:split_idx], 'test': wav_files[split_idx:]}

    width = 40
    encoder = createEncoder(width=width)

    model = Region(
        Layer(din=(20, width), dout=(30, 30)),
        Layer(din=(30, 30), dout=(20, 20))
    )
    model.compile()

    clf = createClassifier()

    speakers = 'm0001 f0002'.split()
    speakers_dict = {k: v for v, k in enumerate(speakers)}

    i = 0
    for phase in ['train', 'test']:
        wavs = dataset[phase]

        for wav in wavs:
            answer, prediction, anomaly = [], [], []

            x, fs = sf.read(wav)
            FEATURES = pw.wav2world(x, fs)
            features = [(f0, sp, ap) for f0, sp, ap in zip(*FEATURES)]

            fig = plt.figure()
            axes, plots = [], []
            for i in range(2):
                data = np.zeros_like(features[0][i+1])
                tmp_ax = fig.add_subplot(2, 1, i+1)
                tmp_ax.set_ylim(FEATURES[i+1].min(), FEATURES[i+1].max())
                plots.append(tmp_ax.plot(data)[0])
                axes.append(tmp_ax)

            ani = anime.FuncAnimation(fig, update, features, interval=50, repeat_delay=1000)
            filename = os.path.basename(wav).replace('.wav', '')
            ani.save(filename + '.gif', writer="imagemagick")
            print(wav)

                # encoding = get_dense_array(mel, encoder, width=width)
                # outputs = model.forward(encoding)
                # output = outputs[-1][0]
                # anomaly.append(model.anomaly())
                #
                # outputs = list(itertools.chain.from_iterable(outputs))
                # viz_util.visualize(i, wav, encoding, outputs)
                #
                # ans = 0
                # for speaker in speakers_dict.keys():
                #     if speaker in wav:
                #         ans = speakers_dict[speaker]
                #
                # if phase == 'train':
                #     clf.learn(output, ans)
                # elif phase == 'test':
                #     pred = np.argmax(clf.infer(output))
                #     answer.append(ans)
                #     prediction.append(pred)
                #
                # i += 1

            # if phase == 'test':
            #     print('answer:', answer)
            #     print('prediction:', prediction)
            #     print('anomaly:', anomaly)
            #     print('accuracy:', np.sum(np.array(answer)==np.array(prediction)))