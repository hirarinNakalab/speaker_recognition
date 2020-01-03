from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.sdr import SDR
from collections import defaultdict
from nnmnkwii.preprocessing import trim_zeros_frames
from sklearn.metrics import f1_score, confusion_matrix

import os
import random
import pysptk
import torchaudio
import numpy as np
import pyworld as pw

from layers import Layer


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


class Experiment:
    def __init__(self, encoder, sdr_length):
        self.encoder = encoder
        self.sdr_length = sdr_length

    def get_encoding(self, feature):
        encodings = [self.encoder.encode(feat) for feat in feature]
        encoding = SDR(self.sdr_length)
        encoding.concatenate(encodings)
        return encoding

    def execute(self, data, model):
        print("wavefile:{}".format(os.path.basename(data)))

        x, fs = torchaudio.load(data)
        x = normalize(x).numpy().reshape(-1).astype(np.float64)

        f0, mcep, bap = get_features(x, fs)
        features = np.concatenate([f0.reshape(-1, 1), mcep[:, :13], -bap], axis=1)

        anomaly = []
        for feature in features:
            encoding = self.get_encoding(feature)
            model.forward(encoding)
            anomaly.append(model.anomaly())

        print("average anomaly score:", np.mean(anomaly), end='\n\n')
        return np.mean(anomaly)


class OVRClassifier:
    def __init__(self, models, sp2idx, experiment):
        self.threshold = 0
        self.models = models
        self.sp2idx = sp2idx
        self.exp = experiment

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
                all_anoms.add(self.exp.execute(data, model))

        results = defaultdict(float)
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
            anomalies[speaker] = self.exp.execute(data, model)
        anom_sorted = sorted(anomalies.items(), key=lambda x: x[1], reverse=True)

        if all([(val > self.threshold) for val in anomalies.values()]):
            return self.sp2idx["unk"]
        else:
            return self.sp2idx[anom_sorted[0].key()]

    def score(self, test_data):
        ans = [self.get_speaker_idx(data) for data in test_data]
        pred = [self.predict(data) for data in test_data]
        data_pair = (ans, pred)
        return f1_score(*data_pair), confusion_matrix(*data_pair)


class Learner:
    def __init__(self, input_path, setting):
        self.split_ratio = 0.8
        self.input_path = input_path
        self.setting = setting
        self.sdr_length = setting["enc"]["size"] * setting["enc"]["featureCount"]
        self.sp2idx = self.speakers_to_idx()
        self.idx2sp = self.idx_to_speakers()
        self.encoder = self.create_encoder()
        self.experiment = self.create_experiment()
        self.train_dataset, self.test_dataset = self.create_dataset()
        self.models = {sp: self.create_model()
                       for sp in self.sp2idx.keys() if not sp == "unk"}
        self.clf = self.create_clf()

    def speakers_to_idx(self):
        speakers = ["unk"] + os.listdir(self.input_path)
        return {k: v for v, k in enumerate(speakers)}

    def idx_to_speakers(self):
        return {k: v for v, k in self.sp2idx.items()}

    def create_dataset(self):
        wav_files = get_wavfile_list(self.input_path)

        speakers_data = defaultdict(list)
        for speaker in self.sp2idx.keys():
            if speaker == "unk":
                continue
            speakers_data[speaker] = [wav for wav in wav_files if speaker in wav]

        sorted_spdata = sorted(speakers_data.items(), key=lambda x: len(x[1]))
        min_length = len(sorted_spdata[0][1])
        split_idx = int(min_length * self.split_ratio)

        train_dataset = defaultdict(list)
        test_dataset = defaultdict(list)

        for speaker in self.sp2idx.keys():
            data = speakers_data[speaker]
            train_dataset[speaker] = data[:split_idx]
            test_dataset[speaker] = data[split_idx:min_length]

        return train_dataset, test_dataset

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
            dout=(output_size,),
            setting=self.setting
        )
        model.compile()
        print()
        return model

    def create_clf(self):
        return OVRClassifier(self.models, self.sp2idx, self.experiment)

    def create_experiment(self):
        return Experiment(self.encoder, self.sdr_length)

    def fit(self, epoch):
        print("=====training phase=====")

        for speaker in self.sp2idx.keys():
            if speaker == "unk":
                continue

            print("=" * 30 + "model of ", speaker, "=" * 30 + "\n")
            model = self.models[speaker]
            model.train()

            train_data = self.train_dataset[speaker]

            for epoch in range(epoch):
                print("epoch {}".format(epoch))
                for data in random.shuffle(train_data):
                    self.exp.execute(data, model)

            print("training data count: {}".format(len(train_data)), end='\n\n')

        all_train_data = [data
                      for speaker in self.sp2idx
                      for data in self.train_dataset[speaker]]

        self.clf.optimize(all_train_data)

    def evaluate(self):
        print("=====training phase=====")

        all_test_data = [data
                          for speaker in self.sp2idx
                          for data in self.test_dataset[speaker]]
        f1, cm = self.clf.score(all_test_data)
        print("testing data count: {}".format(len(all_test_data)), end='\n\n')
        return f1, cm