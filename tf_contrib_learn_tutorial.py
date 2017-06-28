import numpy as np
import tensorflow as tf
# Deactivating warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# This document uses the tutorial codes from TensorFlow website:
# https://www.tensorflow.org/get_started/get_started
# It was rewritten following PEP8 (except 79 char -> 119 char for PyCharm) with additional notes in between the snippets

# tf.contrib.learn: high-level library with to simplify ML procedure


def main():
    basic()
    custom()
    return


def basic():
    # Declare list of features. We only have one real-valued feature. There are many
    # other types of columns that are more complicated and useful.
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

    # An estimator is the front end to invoke training (fitting) and evaluation
    # (inference). There are many predefined types like linear regression,
    # logistic regression, linear classification, logistic classification, and
    # many neural network classifiers and regressors. The following code
    # provides an estimator that does linear regression.

    # The given code causes error on estimator.fit() without specifying model_dir on the line below.
    # https://github.com/tensorflow/tensorflow/issues/7841
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features, model_dir='./output/basic')

    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use two data sets: one for training and one for evaluation
    # We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train,
                                                  batch_size=4,
                                                  num_epochs=1000)
    eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
        {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

    # We can invoke 1000 training steps by invoking the method and passing the
    # training data set.
    estimator.fit(input_fn=input_fn, steps=1000)

    # Here we evaluate how well our model did.
    train_loss = estimator.evaluate(input_fn=input_fn)
    eval_loss = estimator.evaluate(input_fn=eval_input_fn)
    print("train loss: %r"% train_loss)
    print("eval loss: %r"% eval_loss)
    return


def custom():
    # Declare list of features, we only have one real-valued feature
    def model(features, labels, mode):
        # Build a linear model and predict values
        w = tf.get_variable("w", [1], dtype=tf.float64)
        b = tf.get_variable("b", [1], dtype=tf.float64)
        y = w*features['x'] + b
        # Loss sub-graph
        loss = tf.reduce_sum(tf.square(y - labels))
        # Training sub-graph
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss),
                         tf.assign_add(global_step, 1))
        # ModelFnOps connects subgraphs we built to the
        # appropriate functionality.
        return tf.contrib.learn.ModelFnOps(
            mode=mode, predictions=y,
            loss=loss,
            train_op=train)

    # similar to basic: model_dir must be specified
    estimator = tf.contrib.learn.Estimator(model_fn=model, model_dir="./output/custom")
    # define our data sets
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000)
    eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)
    # train
    estimator.fit(input_fn=input_fn, steps=1000)

    # Here we evaluate how well our model did.
    train_loss = estimator.evaluate(input_fn=input_fn)
    eval_loss = estimator.evaluate(input_fn=eval_input_fn)
    print("train loss: %r" % train_loss)
    print("eval loss: %r" % eval_loss)
    return

if __name__ == '__main__':
    main()
