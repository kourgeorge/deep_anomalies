import auto_encoder.network as network
import tensorflow as tf
import utils
import numpy as np

num_epochs = 100
batch_size = 1000
model_path = "./model/model.ckpt"

saver = tf.train.Saver()

label = 8
mnist_train_indices = utils.get_image_indices_of_label(label)[0]
mnist_test_indices = utils.get_image_indices_of_label(label)[1]

with tf.Session() as sess:
    sess.run(network.init)
    num_iterations = int(utils.mnist.train.num_examples / batch_size)

    for epoch in range(num_epochs):
        ep_error = 0
        for i in range(num_iterations):
            xs, ys = utils.mnist2.train.next_batch(batch_size)

            indices = np.where(ys == label)[0]
            xs = xs[indices]

            sess.run(network.opt, feed_dict={network.X: xs})

        test_error, summary, cons_error, l2_regul_error = sess.run(
            [network.cost, network.merged_summary_op, network.cons_error, network.l2_regul_error],
            feed_dict={network.X: utils.mnist2.test.images[mnist_test_indices]})
        train_error = sess.run(network.cost,
                               feed_dict={network.X: utils.mnist2.train.images[mnist_train_indices]})

        # Write logs at every iteration
        network.summary_writer.add_summary(summary, epoch)

        print("Epoch: ", str(epoch), " Training error: ", train_error / len(mnist_train_indices), " Testing error: ",
              test_error / len(mnist_test_indices), " Cons error: ", cons_error, " l2_regularization error: ",
              l2_regul_error)

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
