import tensorflow as tf

n_input = 784
n_hidden1 = 256
n_hidden2 = 50

logs_path = '/tmp/tensorflow_logs/example'

X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])

weights = {
    "encoder1": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_input, n_hidden1])),
    "encoder2": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_hidden1, n_hidden2])),
    "decoder1": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_hidden2, n_hidden1])),
    "decoder2": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_hidden1, n_input])),
}

biases = {
    "encoder1": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_hidden1])),
    "encoder2": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_hidden2])),
    "decoder1": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_hidden1])),
    "decoder2": tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([n_input])),
}

encoder1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["encoder1"]), biases["encoder1"]))
encoder2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder1, weights["encoder2"]), biases["encoder2"]))

decoder1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder2, weights["decoder1"]), biases["decoder1"]))
decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, weights["decoder2"]), biases["decoder2"]))

alpha = 0.01
# cost = tf.reduce_mean(tf.reduce_sum(tf.pow(X - decoder2, 2), reduction_indices=1))
cons_error = tf.nn.l2_loss(X - decoder2)
l2_regul_error = alpha * (tf.nn.l2_loss(weights["encoder1"]) + tf.nn.l2_loss(
    weights["encoder2"]) + tf.nn.l2_loss(weights["decoder1"]) + tf.nn.l2_loss(weights["decoder2"]))
cost = cons_error + l2_regul_error

opt = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", cost)

merged_summary_op = tf.merge_all_summaries()

init = tf.initialize_all_variables()

# op to write logs to Tensorboard
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
