import tensorflow as tf
import utils
import numpy as np


from auto_encoder import network


model_path = "./model/model.ckpt"
images_path = "./out/"

def main():
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        dist = tf.reduce_sum(tf.pow(network.X - network.decoder2, 2), reduction_indices=1)


        cost_array, decoded = sess.run([dist, network.decoder2], feed_dict={network.X: utils.mnist.test.images})

        #utils.draw_top_least_anom(cost_array, 5, images_path)
        #utils.draw_riddle(cost_array, 5, 4, images_path)
        # utils.compare_anom(cost_array, images_path)

        #utils.draw_cost_function(cost_array, images_path)
        #utils.draw_top_bottom_images(utils.mnist.test.images[test_indices], cost_array, 4, images_path)
        #utils.draw_diffs(cost_array, utils.mnist.test.images, decoded, images_path)
        utils.draw_images_to_doc(utils.mnist.test.images)


main()
