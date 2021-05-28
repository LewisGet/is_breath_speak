import config
import numpy as np
import pyworld as pw


def world_decompose(wav, fs, frame_period=config.frame_period):
    wav = wav.astype(np.float64)
    f0, timeaxis = pw.harvest(wav, fs, frame_period = frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pw.cheaptrick(wav, f0, timeaxis, fs)
    ap = pw.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=config.num_mcep):
    coded_sp = pw.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def data_split(dataset, max_len=config.max_wav_size, n_frames=config.n_frames):
    return_dataset = list()
    for data in dataset:
        for i in range(len(data)):
            start = int(i * max_len * 0.5)
            end = start + 1 * max_len
            tmp_data = data[start: end]

            # todo: this is lazy fixed, fixed it perfect
            if len(tmp_data) < max_len:
                tmp_data = data[-1 * max_len:]
                return_dataset.append(tmp_data)
                break

            return_dataset.append(tmp_data)

    return return_dataset
