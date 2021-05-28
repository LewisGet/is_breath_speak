import os

# todo: 之後再區分雜音
# name_lists = ['breath', 'electron', 'normal']
name_lists = ['breath', 'normal', 'none']
data_path = os.path.join(".", "data")
test_path = os.path.join(".", "test", "org")
test_cache_path = os.path.join(".", "test", "cache")
output_path = os.path.join(".", "output")

sampling_rate = 16000
num_mcep = 24
frame_period = 5.0
n_frames = 32
learning_rate = 0.05
max_wav_size = num_mcep * 100
labels = len(name_lists)
size = 31
model_name = "model"

name2id = {char: idx for idx, char in enumerate(name_lists)}
id2name = {idx: char for idx, char in enumerate(name_lists)}
