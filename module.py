import tensorflow as tf
import config

input_shape = [None, config.num_mcep, config.size, 1]
input_X = tf.placeholder(tf.float32, shape=input_shape)
input_Y = tf.placeholder(tf.float32, [None, config.labels])

layers = list()

layer1 = tf.layers.conv2d(
    inputs=input_X,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h1_x"
)

weight1 = tf.layers.conv2d(
    inputs=input_X,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h1_w"
)

layer_weight_1 = tf.multiply(x=layer1, y=tf.sigmoid(weight1), name="h1_wx")

layer2 = tf.layers.conv2d(
    inputs=layer_weight_1,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h2_x"
)

# todo: 詳細了解目的
layer2_norm = tf.contrib.layers.instance_norm(
    inputs=layer2,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)


weight2 = tf.layers.conv2d(
    inputs=layer_weight_1,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h2_w"
)

weight2_norm = tf.contrib.layers.instance_norm(
    inputs=weight2,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)

layer_weight_2 = tf.multiply(x=layer2_norm, y=tf.sigmoid(weight2_norm), name="h2_wx")

layer3 = tf.layers.conv2d(
    inputs=layer_weight_2,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h3_x"
)

# todo: 詳細了解目的
layer3_norm = tf.contrib.layers.instance_norm(
    inputs=layer3,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)


weight3 = tf.layers.conv2d(
    inputs=layer_weight_2,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h3_w"
)

weight3_norm = tf.contrib.layers.instance_norm(
    inputs=weight3,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)

layer_weight_3 = tf.multiply(x=layer3_norm, y=tf.sigmoid(weight3_norm), name="h3_wx")

layer4 = tf.layers.conv2d(
    inputs=layer_weight_3,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h4_x"
)

# todo: 詳細了解目的
layer4_norm = tf.contrib.layers.instance_norm(
    inputs=layer4,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)


weight4 = tf.layers.conv2d(
    inputs=layer_weight_3,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h4_w"
)

weight4_norm = tf.contrib.layers.instance_norm(
    inputs=weight4,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)

layer_weight_4 = tf.multiply(x=layer4_norm, y=tf.sigmoid(weight4_norm), name="h4_wx")

layer5 = tf.layers.conv2d(
    inputs=layer_weight_4,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h5_x"
)

# todo: 詳細了解目的
layer5_norm = tf.contrib.layers.instance_norm(
    inputs=layer5,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)


weight5 = tf.layers.conv2d(
    inputs=layer_weight_4,
    filters=128,
    kernel_size=[3, 3],
    strides=[1, 1],
    activation=None,
    name="h5_w"
)

weight5_norm = tf.contrib.layers.instance_norm(
    inputs=weight5,
    # todo: 詳細了解［目前了解為：接近這小數點的話視同為0 來省略計算］
    epsilon=1e-06,
    activation_fn=None
)

layer_weight_5 = tf.multiply(x=layer5_norm, y=tf.sigmoid(weight5_norm), name="h5_wx")

o1 = tf.layers.dense(inputs=layer_weight_5, units=config.labels, activation=tf.nn.sigmoid)


_f = o1
f = input_Y

loss = tf.reduce_mean(tf.square(f - _f))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
