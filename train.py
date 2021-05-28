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

x = x.reshape(len(x), config.num_mcep * config.size)

feed_indexs = np.arange(len(x))

for i in range(5000):
    np.random.shuffle(feed_indexs)
    sess.run(optimizer, feed_dict={input_X: x[feed_indexs[:]], input_Y: y[feed_indexs[:]]})

print(sess.run(loss, feed_dict={input_X: [x[feed_indexs[0]]], input_Y: [y[feed_indexs[0]]]}))
saver.save(sess, config.model_name)
