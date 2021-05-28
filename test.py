from pydub import AudioSegment
import librosa
import config
import glob
import os
from unit import *
from module import *

def to_wav(new_path, file):
    name = os.path.splitext(os.path.basename(file))[0]

    audio = AudioSegment.from_file(file)

    # , frame_rate=22050, channels=1, sample_width=2

    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    audio.export("%s.wav" % os.path.join(new_path, name), format="wav")


types = ["wav", "mp4", "mp3", "acc", "wma", "m4a"]

files = list()
[files.extend(glob.glob(os.path.join(config.test_path, "*" + i))) for i in types]
[to_wav(config.test_cache_path, i) for i in files]

files = glob.glob(os.path.join(config.test_cache_path, "*.wav"))
wavs = list()
x = list()

for file_path in files:
    wav, _ = librosa.load(file_path, sr=config.sampling_rate, mono=True)
    wavs.append(wav)

wavs = data_split(wavs, config.max_wav_size, config.n_frames)

for wav in wavs:
    f0, timeaxis, sp, ap = world_decompose(wav, config.sampling_rate, config.frame_period)
    tmp_x = world_encode_spectral_envelop(sp, config.sampling_rate, config.num_mcep).T
    x.append(tmp_x)

x = np.array(x)
saver.restore(sess, config.model_name)

x = x.reshape(len(x), config.num_mcep * config.size)
y_ = sess.run(layer_3, feed_dict={input_X: x})

i = 0
for wav, guess in zip(wavs, y_):
    guess_index = np.argmin(guess)
    print(guess_index)

    save_path = os.path.join(config.output_path, config.id2name[guess_index])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    librosa.output.write_wav(os.path.join(save_path, str(i) + ".wav"), wav, config.sampling_rate)
    i += 1
