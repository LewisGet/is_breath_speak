import os
import glob
import librosa
import numpy as np
import pyworld as pw
import tensorflow as tf

import config
from unit import *

#load cache data
x, y = np.load("x.npy"), np.load("y.npy")