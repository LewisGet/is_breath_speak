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
np.random.shuffle(feed_indexs)

for i in range(15000):
    for b in range(len(x)):
        sess.run(optimizer, feed_dict={input_X: [x[feed_indexs[b]]], input_Y: [y[feed_indexs[b]]]})

print(sess.run(loss, feed_dict={input_X: [x[feed_indexs[0]]], input_Y: [y[feed_indexs[0]]]}))
saver.save(sess, config.model_name)
