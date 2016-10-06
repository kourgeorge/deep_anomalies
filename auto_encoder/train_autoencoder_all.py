import auto_encoder.network as network
import tensorflow as tf
import utils
import numpy as np

num_epochs = 30
batch_size = 100
model_path = "./model/model.ckpt"

saver = tf.train.Saver()



with tf.Session() as sess:
    sess.run(network.init)
    num_iterations = int(utils.mnist.train.num_examples / batch_size)

    for epoch in range(num_epochs):
        ep_error = 0
        for i in range(num_iterations):
            xs, ys = utils.mnist2.train.next_batch(batch_size)

            sess.run(network.opt, feed_dict={network.X: xs})

        test_error, summary, cons_error, l2_regul_error = sess.run(
            [network.cost, network.merged_summary_op, network.cons_error, network.l2_regul_error],
            feed_dict={network.X: utils.mnist2.test.images})
        train_error = sess.run(network.cost,
                               feed_dict={network.X: utils.mnist2.train.images})

        # Write logs at every iteration
        network.summary_writer.add_summary(summary, epoch)

        print("Epoch: ", str(epoch), " Training error: ", train_error / utils.mnist.train.num_examples, " Testing error: ",
              test_error / utils.mnist.test.num_examples, " Cons error: ", cons_error, " l2_regularization error: ",
              l2_regul_error)

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
