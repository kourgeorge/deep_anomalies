import tensorflow as tf
import utils

from classification import network

num_epochs = 10
batch_size = 100
model_path = "./model/model.ckpt"


init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
        avg_cost = 0
        num_batches = int(utils.mnist.train.num_examples / batch_size)
        for i in range(num_batches):
            train_x, train_y = utils.mnist.train.next_batch(batch_size)
            sess.run(network.opt, feed_dict={network.x: train_x, network.y: train_y})

        matches = tf.equal(tf.arg_max(network.pred, dimension=1), tf.arg_max(network.y, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(matches, "float"), reduction_indices=0)

        acc = sess.run(accuracy, feed_dict={network.x: utils.mnist.test.images, network.y: utils.mnist.test.labels})
        print("Epoch: ", str(epoch), " accuracy: ", str(acc))

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
