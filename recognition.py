import os
import wave
import itertools

from sklearn.preprocessing import MinMaxScaler
import librosa
import numpy as np
import matplotlib.pyplot as plt

from layers import Region, Layer, createEncoder, createClassifier
from sdr_util import getDenseArray
import viz_util
import param


def get_wavfile_list(path):
    wav_files = []
    for dirpath, subdirs, files in os.walk(path):
        for x in files:
            if x.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, x))
    return wav_files

def get_wave_data(filename, print_info=False):
    with wave.open(filename , "r") as wf:
        fs = wf.getframerate()  # サンプリング周波数
        sig = wf.readframes(wf.getnframes())
        sig = np.frombuffer(sig, dtype= "int16") / 32768.0  # -1 - +1に正規化
        if print_info:
            print_wave_info(wf)
    return sig, fs

def print_wave_info(wf):
    """WAVEファイルの情報を取得"""
    print ("チャンネル数:", wf.getnchannels())
    print ("サンプル幅:", wf.getsampwidth())
    print ("サンプリング周波数:", wf.getframerate())
    print ("フレーム数:", wf.getnframes())
    print ("パラメータ:", wf.getparams())
    print ("長さ（秒）:", float(wf.getnframes()) / wf.getframerate())

def normalize_sig(sig):
    scaler = MinMaxScaler()
    sig = sig.reshape(-1, 1)
    sig = scaler.fit_transform(sig)
    return sig.reshape(-1)

def normalize_spec(spec, min_level_db=-100):
    return np.clip((spec - min_level_db) / -min_level_db, 0, 1)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def build_mel_basis():
    args = {
        'sr': fs,
        'n_fft': 2048,
        'n_mels': 20,
        'fmin': 40
    }
    return librosa.filters.mel(**args)

def linear_to_mel(spectrogram, mel_basis=None):
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def mel_spectrogram(sig, fs):
    args = {
        'y': sig,
        'n_fft': 2048,
        'hop_length': int(fs * 0.0125), # 12.5ms
        'win_length': int(fs * 0.05)   # 50ms
    }
    spectrum = librosa.stft(**args)
    db = amp_to_db(linear_to_mel(np.abs(spectrum)))
    return normalize_spec(db)

def show_spectrogram(mel, save=False):
    plt.imshow(mel)
    plt.show()
    if save:
        plt.savefig('mel_spectrogram.png')

if __name__ == '__main__':
    data_path = param.input_file
    wav_files = get_wavfile_list(data_path)

    width = 40
    encoder = createEncoder(width=width)

    model = Region(
        Layer(din=(20, width), dout=(20, 20), temporal=True),
        Layer(din=(20, 20), dout=(10, 10), temporal=False)
    )
    model.compile()

    clf = createClassifier()

    speaker_dict = {
        'm0001': 1,
        'f0002': 2
    }

    i = 0
    for wav_file in wav_files:
        answer, prediction = [], []

        sig, fs = get_wave_data(wav_file)
        sig = normalize_sig(sig)
        mels = mel_spectrogram(sig, fs)

        for mel in mels.T:
            encoding = getDenseArray(mel, encoder, width=width)
            outputs = model.foward(encoding)
            output = outputs[-1][0]

            outputs = list(itertools.chain.from_iterable(outputs))
            outputs = [output for output in outputs if output is not None]

            viz_util.visualize(i, wav_file, encoding, outputs)

            ans = 0
            for speaker in speaker_dict.keys():
                if speaker in wav_file:
                    ans = speaker_dict[speaker]

            if 'test' in wav_file:
                pred = np.argmax(clf.infer(output))
                answer.append(ans)
                prediction.append(pred)
            else:
                clf.learn(output, ans)

            i += 1

        print(wav_file)

        if 'test' in wav_file:
            print('answer:', answer)
            print('prediction:', prediction)
            print('accuracy:', np.sum(np.array(answer)==np.array(prediction)))