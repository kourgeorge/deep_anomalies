import tensorflow as tf
from classification import network
import utils

model_path = "./model/model.ckpt"
images_path = "./out/"

def main():
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        load_path = saver.restore(sess, model_path)

        #dist = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(network.pred, network.y), reduction_indices=1)
        dist = tf.nn.softmax_cross_entropy_with_logits(network.pred, network.y)
        cost_array = sess.run(dist, feed_dict={network.x: utils.mnist.test.images, network.y: utils.mnist.test.labels})

        # utils.draw_top_least_anom(cost_array, 5, images_path)
        #utils.draw_riddle(cost_array, 5, 4, images_path)
        utils.compare_anom(cost_array, images_path)


main()

