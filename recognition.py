import os
import itertools

import pyworld as pw
import soundfile as sf
import numpy as np

from layers import Region, Layer, create_encoder, create_classifier
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
    clf = create_classifier()
    encoder = create_encoder(width=width)
    model = Region(
        Layer(din=(20, width), dout=(30, 30)),
        Layer(din=(30, 30), dout=(20, 20))
    )
    model.compile()
    return encoder, clf, model

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

def get_features_iterator(features):
    return [(f0, sp, ap) for f0, sp, ap in zip(*features)]


if __name__ == '__main__':
    dataset = create_dataset(data_path = param.input_file)
    encoder, clf, model = define_model(width=40)

    speakers = 'm0001 f0002'
    speakers_dict = create_speakers_dict(speakers)

    i = 0
    for phase in ['train', 'test']:
        wavs = dataset[phase]

        for wav in wavs:
            answer, prediction, anomaly = [], [], []

            x, fs = sf.read(wav)
            features = pw.wav2world(x, fs)
            features_iter = get_features_iterator(features)

            write_gif_file(features, features_iter, filename=wav)

            ans = get_speaker_idx(speakers_dict, wav)
            for f0, sp, ap in features_iter:

                encoding = get_dense_array(mel, encoder, width=width)
                outputs = model.forward(encoding)
                output = outputs[-1][0]
                anomaly.append(model.anomaly())

                outputs = list(itertools.chain.from_iterable(outputs))
                viz_util.visualize(i, wav, encoding, outputs)

                if phase == 'train':
                    clf.learn(output, ans)
                elif phase == 'test':
                    pred = np.argmax(clf.infer(output))
                    answer.append(ans)
                    prediction.append(pred)

                i += 1

            if phase == 'test':
                print('answer:', answer)
                print('prediction:', prediction)
                print('anomaly:', anomaly)
                print('accuracy:', np.sum(np.array(answer)==np.array(prediction)))