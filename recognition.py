import os

from nnmnkwii.preprocessing import trim_zeros_frames
import numpy as np
import pyworld as pw
import pysptk
import torchaudio

from layers import create_encoder, create_model
from sdr_util import get_encoding
import param


def create_dataset(data_path, speakers_dict):
    wav_files = get_wavfile_list(data_path)
    speakers_data = {speaker: [wav for wav in wav_files if speaker in wav]
                     for speaker in speakers_dict.keys()}
    dataset = {'train': dict(), 'test':dict()}
    for phase in ['train', 'test']:
        for speaker in speakers_dict.keys():
            data = speakers_data[speaker]
            split_idx = int(len(data) * 0.6)
            if phase == "train":
                dataset[phase][speaker] = data[:split_idx]
            elif phase == "test":
                dataset[phase][speaker] = data[split_idx:]
    return dataset

def create_speakers_dict(speakers):
    speakers = speakers.split()
    return {k: v for v, k in enumerate(speakers)}

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

def get_length_dict(dataset, speakers_dict):
    length_dict = {}
    for phase in ["train", "test"]:
        length = 100
        for speaker in speakers_dict.keys():
            length = min(length, len(dataset[phase][speaker]))
        length_dict[phase] = length
    return length_dict

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

def experiment(wavs, encoder, model):
    for wav in wavs:
        print("wavefile:{}".format(wav))

        x, fs = torchaudio.load(wav)
        x = normalize(x).numpy().reshape(-1).astype(np.float64)

        f0, mcep, bap = get_features(x, fs)
        features = np.concatenate([f0.reshape(-1, 1), mcep[:, :13], -bap], axis=1)

        anomaly = []
        for feature in features:
            encoding = get_encoding(encoder, feature)
            model.forward(encoding)
            anomaly.append(model.anomaly())

        print("average anomaly score:", np.mean(anomaly), end='\n\n')

def main():

    speakers = 'm0001 f0002'
    speakers_dict = create_speakers_dict(speakers)

    dataset = create_dataset(param.input_file, speakers_dict)
    encoder = create_encoder()

    length_dict = get_length_dict(dataset, speakers_dict)

    models = {}
    for phase in ['train', 'test']:
        print("====={}ing phase=====".format(phase))
        print()
        length = length_dict[phase]

        for speaker in speakers_dict.keys():
            print("="*30)
            print("model of ", speaker)
            print("="*30)
            print()

            if phase == "train":
                model = create_model()
                models[speaker] = model

                model.train()

                wav_data = dataset[phase][speaker][:length]
            else:
                model = models[speaker]
                model.eval()

                wav_data = [data
                             for speaker in speakers_dict.keys()
                             for data in dataset[phase][speaker][:length]]

            experiment(wav_data, encoder, model)
            print("{}ing data count: {}".format(phase, len(wav_data)), end='\n\n')


if __name__ == '__main__':
    main()