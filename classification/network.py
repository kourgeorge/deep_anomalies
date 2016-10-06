import tensorflow as tf


input_size = 784
num_classes = 10
h1_size = 256
h2_size = 256

learning_rate = 0.005


x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

# define the variables
weight = {
    "h1": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[input_size, h1_size])),
    "h2": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[h1_size, h2_size])),
    "out": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[h2_size, num_classes]))
}

bias = {
    "h1": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[h1_size])),
    "h2": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[h2_size])),
    "out": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[num_classes]))
}

# build up the graph connectivity
h1 = tf.nn.relu(tf.add(tf.matmul(x, weight["h1"]), bias["h1"]))
h1 = tf.nn.dropout(h1, 0.75)
h2 = tf.nn.relu(tf.add(tf.matmul(h1, weight["h2"]), bias["h2"]))
pred = tf.add(tf.matmul(h1, weight["out"]), bias["out"])

# calculate the cost function and the optimization algorithm
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
