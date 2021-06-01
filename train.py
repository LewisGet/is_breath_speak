import os
import glob
import librosa
import numpy as np
import pyworld as pw
import tensorflow as tf

import config
from unit import *
from module import *

#load cache data
x, y = np.load("x.npy"), np.load("y.npy")

print(x.shape, y.shape)

x = x.reshape(len(x), config.num_mcep, config.size, 1)

feed_indexs = np.arange(len(x))
max_feed = int(len(x) / 4)

for i in range(10):
    np.random.shuffle(feed_indexs)
    feed_x = x[feed_indexs[:max_feed]]
    feed_y = y[feed_indexs[:max_feed]]
    feed_in = {input_X: feed_x, input_Y: feed_y}
    sess.run(optimizer, feed_dict=feed_in)

    print(sess.run(loss, feed_dict=feed_in))

print(sess.run(loss, feed_dict=feed_in))
saver.save(sess, config.model_name)