import os
import itertools

from nnmnkwii.preprocessing import trim_zeros_frames
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import pysptk
import torchaudio

from layers import Region, Layer, create_encoder
from sdr_util import get_dense_array
from viz_util import visualize, write_gif_file
import param


def create_dataset(data_path):
    wav_files = get_wavfile_list(data_path)
    n_files = len(wav_files)
    split_idx = int(n_files * 0.6)
    dataset = {'train': wav_files[:split_idx], 'test': wav_files[split_idx:]}
    return dataset

def create_speakers_dict(speakers):
    speakers = speakers.split()
    return {k: v for v, k in enumerate(speakers)}

def define_model(width):
    encoder = create_encoder(width=width)
    model = Region(
        Layer(din=(20, width), dout=(30, 30)),
        Layer(din=(30, 30), dout=(20, 20))
    )
    model.compile()
    return encoder, model

def get_wavfile_list(path):
    wav_files = []
    for dirpath, subdirs, files in os.walk(path):
        for x in files:
            if x.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, x))
    return wav_files

def get_speaker_idx(speakers_dict, filename):
    ans = 0
    for speaker in speakers_dict.keys():
        if speaker in filename:
            ans = speakers_dict[speaker]
    return ans

def get_features(x, fs):
    # f0 calculate
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    # mcep calculate
    sp = trim_zeros_frames(pw.cheaptrick(x, f0, t, fs))
    mcep = pysptk.sp2mc(sp, order=24, alpha=pysptk.util.mcepalpha(fs))
    # bap calculate
    ap = pw.d4c(x, f0, t, fs)
    bap = pw.code_aperiodicity(ap, fs)
    return f0, mcep, bap

def get_features_iterator(features):
    return [(f0, sp, ap) for f0, sp, ap in zip(*features)]

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()



if __name__ == '__main__':

    dataset = create_dataset(data_path = param.input_file)
    encoder, model = define_model(width=40)

    speakers = 'm0001 f0002'
    speakers_dict = create_speakers_dict(speakers)

    fig, axes = plt.subplots(16, 1)
    for phase in ['train', 'test']:
        wavs = dataset[phase]

        for wav in wavs:
            print(wav)

            answer, prediction, anomaly = [], [], []

            x, fs = torchaudio.load(wav)

            x = normalize(x).numpy().reshape(-1).astype(np.float64)

            f0, mcep, bap = get_features(x, fs)

            features = np.concatenate([f0.reshape(-1, 1), mcep[:, :13], -bap], axis=1)

            axes[0].cla()
            axes[0].plot(x)
            for i, feature in enumerate(features.T, start=1):
                axes[i].cla()
                axes[i].plot(feature)

            plt.pause(1)

            filename = os.path.basename(wav).replace(".wav", "") + ".png"
            plt.savefig(filename)

            # write_gif_file(features, features_iter, filename=wav)

            # ans = get_speaker_idx(speakers_dict, wav)
            #
            # encoding = get_dense_array(mel, encoder, width=width)
            # outputs = model.forward(encoding)
            # anomaly.append(model.anomaly())
            #
            # outputs = list(itertools.chain.from_iterable(outputs))
            # visualize(i, wav, encoding, outputs)