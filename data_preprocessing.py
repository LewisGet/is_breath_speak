import os
import config
import glob
import librosa
from unit import *
import numpy as np


x, y = list(), list()

for clazz in config.name_lists:
    path = os.path.join(config.data_path, clazz, "*.wav")
    file_paths = glob.glob(path)
    wavs = list()

    for file_path in file_paths:
        wav, _ = librosa.load(file_path, sr=config.sampling_rate, mono=True)
        wavs.append(wav)

    wavs = data_split(wavs, config.max_wav_size, config.n_frames)

    for wav in wavs:
        f0, timeaxis, sp, ap = world_decompose(wav, config.sampling_rate, config.frame_period)

        tmp_x = world_encode_spectral_envelop(sp, config.sampling_rate, config.num_mcep).T

        tmp_y = np.zeros(len(config.name_lists))
        tmp_y[config.name2id[clazz]] = 1

        x.append(tmp_x)
        y.append(tmp_y)

np.save("x.npy", x)
np.save("y.npy", y)
