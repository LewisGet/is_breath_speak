import tensorflow as tf
import config

input_shape = [None, config.num_mcep, config.size, 1]
input_X = tf.placeholder(tf.float32, shape=input_shape)
input_Y = tf.placeholder(tf.float32, [None, config.labels])

layers = list()

# (?, 22, 29, 128)
layer1 = tf.layers.conv2d(
    inputs=input_X,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h1_x"
)

# (?, 22, 29, 128)
weight1 = tf.layers.conv2d(
    inputs=input_X,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h1_w"
)

# (?, 22, 29, 128)
layer_weight_1 = tf.multiply(x=layer1, y=tf.sigmoid(weight1), name="h1_wx")

# (?, 22, 15, 128)
layer2 = tf.layers.conv2d(
    inputs=layer_weight_1,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 2],
    padding='same',
    activation=None,
    name="h2_x"
)

# (?, 22, 15, 128)
layer2_norm = tf.contrib.layers.instance_norm(
    inputs=layer2,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 22, 15, 128)
weight2 = tf.layers.conv2d(
    inputs=layer_weight_1,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 2],
    padding='same',
    activation=None,
    name="h2_w"
)

# (?, 22, 15, 128)
weight2_norm = tf.contrib.layers.instance_norm(
    inputs=weight2,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 22, 15, 128)
layer_weight_2 = tf.multiply(x=layer2_norm, y=tf.sigmoid(weight2_norm), name="h2_wx")

# (?, 11, 8, 256)
layer3 = tf.layers.conv2d(
    inputs=layer_weight_2,
    filters=256,
    kernel_size=[3, 3],
    strides=[2, 2],
    padding='same',
    activation=None,
    name="h3_x"
)

# (?, 11, 8, 256)
layer3_norm = tf.contrib.layers.instance_norm(
    inputs=layer3,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 11, 8, 256)
weight3 = tf.layers.conv2d(
    inputs=layer_weight_2,
    filters=256,
    kernel_size=[3, 3],
    strides=[2, 2],
    padding='same',
    activation=None,
    name="h3_w"
)

# (?, 11, 8, 256)
weight3_norm = tf.contrib.layers.instance_norm(
    inputs=weight3,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 11, 8, 256)
layer_weight_3 = tf.multiply(x=layer3_norm, y=tf.sigmoid(weight3_norm), name="h3_wx")

# (?, 11, 8, 256)
layer4 = tf.layers.conv2d(
    inputs=layer_weight_3,
    filters=512,
    kernel_size=[3, 3],
    strides=[3, 3],
    padding='same',
    activation=None,
    name="h4_x"
)

# (?, 11, 8, 256)
layer4_norm = tf.contrib.layers.instance_norm(
    inputs=layer4,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 11, 8, 256)
weight4 = tf.layers.conv2d(
    inputs=layer_weight_3,
    filters=512,
    kernel_size=[3, 3],
    strides=[3, 3],
    padding='same',
    activation=None,
    name="h4_w"
)

# (?, 11, 8, 256)
weight4_norm = tf.contrib.layers.instance_norm(
    inputs=weight4,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 11, 8, 256)
layer_weight_4 = tf.multiply(x=layer4_norm, y=tf.sigmoid(weight4_norm), name="h4_wx")

# (?, 2, 2, 1024)
layer5 = tf.layers.conv2d(
    inputs=layer_weight_4,
    filters=1024,
    kernel_size=[6, 3],
    strides=[2, 2],
    padding='same',
    activation=None,
    name="h5_x"
)

# (?, 2, 2, 1024)
layer5_norm = tf.contrib.layers.instance_norm(
    inputs=layer5,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 2, 2, 1024)
weight5 = tf.layers.conv2d(
    inputs=layer_weight_4,
    filters=1024,
    kernel_size=[6, 3],
    strides=[2, 2],
    padding='same',
    activation=None,
    name="h5_w"
)

# (?, 2, 2, 1024)
weight5_norm = tf.contrib.layers.instance_norm(
    inputs=weight5,
    epsilon=1e-06,
    activation_fn=None
)

# (?, 2, 2, 1024)
layer_weight_5 = tf.multiply(x=layer5_norm, y=tf.sigmoid(weight5_norm), name="h5_wx")

layer_weight_5_flat = tf.contrib.layers.flatten(layer_weight_5)

o1 = tf.layers.dense(inputs=layer_weight_5_flat, units=config.labels, activation=tf.nn.sigmoid)

dropout_1 = tf.layers.dense(layer_weight_5_flat, 2048, activation=tf.nn.relu, name='dropout_1')
dropout_1 = tf.layers.dropout(dropout_1, 0.5, name='dropout_1_execute')
dropout_1 = tf.layers.dense(dropout_1, config.labels, activation=tf.nn.relu, name='dropout_1_f')

dropout_2 = tf.layers.dense(layer_weight_5_flat, 1024, activation=tf.nn.relu, name='dropout_2')
dropout_2 = tf.layers.dropout(dropout_1, 0.5, name='dropout_2_execute')
dropout_2 = tf.layers.dense(layer_weight_5_flat, config.labels, activation=tf.nn.relu, name='dropout_2_f')

f = input_Y

loss1 = tf.reduce_mean(tf.square(f - o1))
loss2 = tf.reduce_mean(tf.square(f - dropout_1))
loss3 = tf.reduce_mean(tf.square(f - dropout_2))
loss = loss1 + loss2 + loss3

optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
