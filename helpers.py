from collections import defaultdict
from sklearn.metrics import f1_score
from htm.encoders.rdse import RDSE, RDSE_Parameters
from nnmnkwii.preprocessing import trim_zeros_frames

import os
import random
import numpy as np
import pyworld as pw
import pysptk
import torchaudio

from sdr_util import get_encoding
from layers import Layer
import param


def get_wavfile_list(path):
    wav_files = []
    for dirpath, subdirs, files in os.walk(path):
        for x in files:
            if x.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, x))
    return wav_files

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

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

def experiment(data, encoder, model, setting):
    print("wavefile:{}".format(os.path.basename(data)))

    x, fs = torchaudio.load(data)
    x = normalize(x).numpy().reshape(-1).astype(np.float64)

    f0, mcep, bap = get_features(x, fs)
    features = np.concatenate([f0.reshape(-1, 1), mcep[:, :13], -bap], axis=1)

    anomaly = []
    for feature in features:
        encoding = get_encoding(encoder, feature, setting)
        model.forward(encoding)
        anomaly.append(model.anomaly())

    print("average anomaly score:", np.mean(anomaly), end='\n\n')
    return np.mean(anomaly)


class OVRClassifier:
    def __init__(self, models, sp2idx, encoder):
        self.threshold = 0
        self.models = models
        self.sp2idx = sp2idx
        self.encoder = encoder

    def get_speaker_idx(self, filename):
        ans = 0
        for speaker in self.sp2idx.keys():
            if speaker in filename:
                ans = self.sp2idx[speaker]
        return ans

    def optimize(self, train_data):
        all_anoms = set()
        for model in self.models.values():
            model.eval()
            for data in train_data:
                all_anoms.add(experiment(data, self.encoder, model))

        results = {}
        for th in sorted(all_anoms, reverse=True):
            self.threshold = th
            ans = [self.get_speaker_idx(data) for data in train_data]
            pred = [self.predict(data) for data in train_data]
            results[th] = f1_score(ans, pred)

        results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
        self.threshold = float(results_sorted[0][1])

    def predict(self, data):
        anomalies = {}
        for speaker in self.sp2idx.keys():
            if speaker == "unk":
                continue
            model = self.models[speaker]
            model.eval()
            anomalies[speaker] = experiment(data, self.encoder, model)
        anom_sorted = sorted(anomalies.items(), key=lambda x: x[1], reverse=True)

        if all([(val > self.threshold) for val in anomalies.values()]):
            return self.sp2idx["unk"]
        else:
            return self.sp2idx[anom_sorted[0].key()]

    def score(self, test_data):
        ans = [self.get_speaker_idx(data) for data in test_data]
        pred = [self.predict(data) for data in test_data]
        return f1_score(ans, pred)


class Learner:
    def __init__(self, path, setting):
        self.setting = setting
        self.sp2idx = self.speakers_to_idx(path)
        self.idx2sp = self.idx_to_speakers()
        self.dataset = self.create_dataset()
        self.length_dict = self.create_length_dict()
        self.encoder = self.create_encoder()
        self.models = {sp: self.create_model()
                       for sp in self.sp2idx.keys() if not sp == "unk"}
        self.clf = self.create_clf()

    def speakers_to_idx(self, path):
        speakers = ["unk"] + os.listdir(path)
        return {k: v for v, k in enumerate(speakers)}

    def idx_to_speakers(self):
        return {k: v for v, k in self.sp2idx.items()}

    def create_dataset(self):
        data_path = param.input_file
        wav_files = get_wavfile_list(data_path)
        speakers_data = {speaker: [wav for wav in wav_files if speaker in wav]
                         for speaker in self.sp2idx.keys()}
        dataset = defaultdict(lambda: defaultdict(list))
        for phase in ['train', 'test']:
            for speaker in self.sp2idx.keys():
                data = speakers_data[speaker]
                split_idx = int(len(data) * 0.8)
                if phase == "train":
                    dataset[phase][speaker] = data[:split_idx]
                elif phase == "test":
                    dataset[phase][speaker] = data[split_idx:]
        return dataset

    def create_length_dict(self):
        length_dict = {}
        for phase in ["train", "test"]:
            length = 100
            for speaker in self.sp2idx.keys():
                length = min(length, len(self.dataset[phase][speaker]))
            length_dict[phase] = length
        return length_dict

    def create_encoder(self):
        print("creating encoder...")
        print(self.setting["enc"])
        scalarEncoderParams = RDSE_Parameters()
        scalarEncoderParams.size = self.setting["enc"]["size"]
        scalarEncoderParams.sparsity = self.setting["enc"]["sparsity"]
        scalarEncoderParams.resolution = self.setting["enc"]["resolution"]
        scalarEncoder = RDSE(scalarEncoderParams)
        print()
        return scalarEncoder

    def create_model(self):
        print("creating model...")
        print(self.setting["sp"])
        print(self.setting["tm"])
        input_size = self.setting["enc"]["size"] * self.setting["enc"]["featureCount"]
        output_size = self.setting["sp"]["columnCount"]
        model = Layer(
            din=(input_size,),
            dout=(output_size,)
        )
        model.compile()
        print()
        return model

    def create_clf(self):
        return OVRClassifier(self.models, self.sp2idx)

    def get_phase_whole_data(self, phase):
        return [data
                for speaker in self.sp2idx.keys()
                for data in self.dataset[phase][speaker][:self.length_dict[phase]]]

    def fit(self, epoch):
        phase = "train"
        for speaker in self.sp2idx.keys():
            if speaker == "unk":
                continue

            print("=" * 30 + "model of ", speaker, "=" * 30 + "\n")
            model = self.models[speaker]
            model.train()

            train_data = self.dataset[phase][speaker][:self.length_dict[phase]]

            for epoch in range(epoch):
                print("epoch {}".format(epoch))
                for data in random.shuffle(train_data):
                    experiment(data, self.encoder, model, self.setting)

            print("{}ing data count: {}".format(phase, len(train_data)), end='\n\n')

        train_data = self.get_phase_whole_data(phase)
        self.clf.optimize(train_data)

    def evaluate(self):
        phase = "test"
        test_data = self.get_phase_whole_data(phase)
        f1 = self.clf.score(test_data)
        print("{}ing data count: {}".format(phase, len(test_data)), end='\n\n')