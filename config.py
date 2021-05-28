import os

# todo: 之後再區分雜音
# name_lists = ['breath', 'electron', 'normal']
name_lists = ['breath', 'normal']
data_path = os.path.join(".", "data")

sampling_rate = 16000
num_mcep = 24
frame_period = 5.0
n_frames = 32
learning_rate = 0.05
max_wav_size = num_mcep * 1000
labels = len(name_lists)

name2id = {char: idx for idx, char in enumerate(name_lists)}
id2name = {idx: char for idx, char in enumerate(name_lists)}
