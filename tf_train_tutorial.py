import tensorflow as tf
# Deactivating warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This document uses the tutorial codes from TensorFlow website:
# https://www.tensorflow.org/get_started/get_started
# It was rewritten following PEP8 (except 79 char -> 119 char for PyCharm) with additional notes in between the snippets

# tf.train: built-in optimizer for variables in respect to a given function (cost function, typ)


def main():
    # Model parameters
    w = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = w * x + b
    y = tf.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # aka "cost function"

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)  # Learning rate lambda = 0.01
    train = optimizer.minimize(loss)  # "Using gradient descent, find weight & bias to minimize loss (cost)"

    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})  # Iterate gradient descent for 1000 times for the given x & y
        if i % 100 == 0:
            # value of weight & bias are improved each iteration (unless an inappropriate learning rate is used)
            print("Iteration #%d: w: %s b: %s" % (i, *sess.run([w, b])))

    # evaluate training accuracy
    curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train})
    print("w: %s b: %s loss: %s" % (curr_w, curr_b, curr_loss))
    return

if __name__ == '__main__':
    main()
