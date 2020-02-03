from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.sdr import SDR
from collections import defaultdict
from nnmnkwii.preprocessing import trim_zeros_frames
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from datetime import datetime
import os
import random
import pysptk
import soundfile as sf
import torchaudio as ta
import numpy as np
import pyworld as pw

from layers import Layer, Unknown
from viz_util import plot_features, plot_input_data, plot_anomalies
from viz_util import plot_waveform, plot_specgram
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

def peak_normalize(data):
    data = data.astype(np.float64)
    amp = max(np.abs(np.max(data)), np.abs(np.min(data)))
    data = data / amp
    data.clip(-1, 1)
    return data

def normalize(tensor):
    tensor_minus_mean = tensor - tensor.mean()
    return tensor_minus_mean / tensor_minus_mean.abs().max()

def sort_dict(dict):
    return sorted(dict.items(), key=lambda x: x[1])

def sort_dict_reverse(dict):
    return sorted(dict.items(), key=lambda x: x[1], reverse=True)

def sort_dict_by_len(dict):
    return sorted(dict.items(), key=lambda x: len(x[1]))


class Experiment:
    def __init__(self, encoder, sdr_length, n_features):
        self.encoder = encoder
        self.sdr_length = sdr_length
        self.n_features = n_features
        self.mel = ta.transforms.MelSpectrogram(n_mels=self.n_features)

    def get_encoding(self, feature):
        encodings = [self.encoder.encode(feat) for feat in feature]
        encoding = SDR(self.sdr_length * self.n_features)
        encoding.concatenate(encodings)
        return encoding

    def get_mel_sp(self, data):
        x, fs = ta.load(data)
        # plot_waveform(x.detach().numpy().reshape(-1))
        features = self.mel(normalize(x)).log2()
        features = features.detach().numpy().astype(np.float32)
        features = features.reshape(features.shape[1], -1)
        # plot_specgram(features)
        return features

    def get_world_features(self, data):
        x, fs = sf.read(data)
        f0, mcep, bap = get_features(x, fs)
        features = np.concatenate([
            f0.reshape(-1, 1),
            mcep[:, :self.n_features - 2],
            -bap
        ], axis=1)
        plot_features(x, features, data, param.default_parameters)
        return features

    def execute(self, data, model):
        print("wavefile:{}".format(os.path.basename(data)))

        features = self.get_mel_sp(data)

        anomaly = []
        for feature in features.T:
            inp = self.get_encoding(feature)
            # plot_input_data(inp)
            act, pred = model.forward(inp)
            anomaly.append(model.anomaly())
        # plot_anomalies(anomaly)
        model.reset()

        score = np.mean(anomaly)
        print("anomaly score:", score, end='\n\n')
        return score

class OVRClassifier:
    def __init__(self, models, sp2idx, experiment, unknown):
        self.threshold = 0
        self.models = models
        self.unknown = unknown
        self.sp2idx = sp2idx
        self.exp = experiment

    def get_speaker_idx(self, filename):
        ans = 0
        for speaker in self.sp2idx.keys():
            if speaker in filename:
                ans = self.sp2idx[speaker]
        return ans

    def optimize(self, train_data):
        all_anoms = defaultdict(lambda: defaultdict(float))
        for data in train_data:
            for model_name, model in self.models.items():
                model.eval()
                all_anoms[data][model_name] = self.exp.execute(data, model)

        anom_patterns = {all_anoms[data][model_name]
                         for data in train_data
                         for model_name in self.models.keys()}

        results = defaultdict(float)
        for th in sorted(anom_patterns, reverse=True):
            ans = [self.get_speaker_idx(data) for data in train_data]
            pred = []
            for data in train_data:
                anoms = all_anoms[data]
                anoms[self.unknown] = th
                anom_sorted = sort_dict(anoms)
                pred_sp = anom_sorted[0][0]
                pred.append(self.sp2idx[pred_sp])
            results[th] = f1_score(ans, pred, average='macro')

        results_sorted = sort_dict_reverse(results)
        print("best score for train data:", results_sorted[0])
        self.models[self.unknown].threshold = float(results_sorted[0][0])

    def predict(self, data):
        anomalies = {}
        for speaker in self.sp2idx.keys():
            model = self.models[speaker]
            model.eval()
            anomalies[speaker] = self.exp.execute(data, model)
        anom_sorted = sort_dict(anomalies)
        pred_sp = anom_sorted[0][0]
        return self.sp2idx[pred_sp]

    def score(self, test_data):
        ans = [self.get_speaker_idx(data) for data in test_data]
        pred = [self.predict(data) for data in test_data]
        data_pair = (ans, pred)

        f1 = f1_score(*data_pair, average="macro")
        cm = confusion_matrix(*data_pair)
        target_names = ["unknown" if target == self.unknown
                        else target for target in self.sp2idx.keys()]
        report = classification_report(*data_pair, target_names=target_names)
        return f1, cm, report

class Learner:
    def __init__(self, input_path, setting, unknown, save_threshold):
        self.setting = setting
        self.split_ratio = self.setting.ratio
        self.input_path = input_path
        self.unknown = unknown
        self.sp2idx = self.speakers_to_idx()
        self.idx2sp = self.idx_to_speakers()
        self.encoder = self.create_encoder()
        self.experiment = self.create_experiment()
        self.train_dataset, self.test_dataset = self.create_dataset()
        self.models = self.create_models()
        self.clf = self.create_clf()
        self.score = 0.0
        self.save_threshold = save_threshold

    def speakers_to_idx(self):
        speakers = os.listdir(self.input_path)
        speakers = [speaker for speaker in speakers
                    if not speaker == self.unknown]
        speakers = [self.unknown] + speakers
        return {k: v for v, k in enumerate(speakers)}

    def idx_to_speakers(self):
        return {k: v for v, k in self.sp2idx.items()}

    def create_dataset(self):
        wav_files = get_wavfile_list(self.input_path)

        speakers_data = defaultdict(list)
        for speaker in self.sp2idx.keys():
            speakers_data[speaker] = [wav for wav in wav_files if speaker in wav]

        sorted_spdata = sort_dict_by_len(speakers_data)
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
        print(self.setting("enc"))
        scalarEncoderParams = RDSE_Parameters()
        scalarEncoderParams.size = self.setting("enc").size
        scalarEncoderParams.sparsity = self.setting("enc").sparsity
        scalarEncoderParams.resolution = self.setting("enc").resolution
        scalarEncoder = RDSE(scalarEncoderParams)
        print()
        return scalarEncoder

    def create_model(self):
        print("creating model...")
        print(self.setting("sp"))
        print(self.setting("tm"))
        input_size = self.setting("enc").size * self.setting("enc").featureCount
        output_size = self.setting("sp").columnCount
        model = Layer(
            din=(input_size,),
            dout=(output_size,),
            setting=self.setting
        )
        model.compile()
        print()
        return model

    def create_clf(self):
        return OVRClassifier(self.models, self.sp2idx, self.experiment, self.unknown)

    def create_experiment(self):
        return Experiment(self.encoder, self.setting("enc").size, self.setting("enc").featureCount)

    def create_models(self):
        d = dict()
        for speaker in self.sp2idx.keys():
            if speaker == self.unknown:
                d[speaker] = Unknown()
            else:
                d[speaker] = self.create_model()
        return d

    def get_all_data(self, dataset):
        return [data
                for speaker in self.sp2idx
                for data in dataset[speaker]]

    def fit(self, epochs):
        print("=====training phase=====")

        for speaker in self.sp2idx.keys():
            print("=" * 30 + "model of ", speaker, "=" * 30 + "\n")
            model = self.models[speaker]
            model.train()

            train_data = self.train_dataset[speaker]

            for epoch in range(epochs):
                print("epoch {}".format(epoch))
                for data in random.sample(train_data, len(train_data)):
                    self.experiment.execute(data, model)

            fmt = "training data count: {}"
            print(fmt.format(len(train_data)), end='\n\n')

        all_train_data = self.get_all_data(self.train_dataset)

        print("=====threshold optimization phase=====")
        self.clf.optimize(all_train_data)

    def evaluate(self):
        print("=====testing phase=====")

        all_test_data = self.get_all_data(self.test_dataset)
        f1, cm, report = self.clf.score(all_test_data)
        self.score = f1

        fmt = "testing data count: {}"
        print(fmt.format(len(all_test_data)), end='\n\n')
        print("test threshold: ", self.models[self.unknown].threshold)
        return f1, cm, report

    def save(self):
        if self.score < self.save_threshold:
            return

        dirname = '-'.join([datetime.now().isoformat(), str(self.score)])
        if os.path.exists(dirname):
            return
        os.mkdir(dirname)

        for speaker, model in self.models.items():
            filename = os.path.join(dirname, speaker)
            model.save(filename)