import tensorflow as tf
# Deactivating warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This document uses the tutorial codes from TensorFlow website:
# https://www.tensorflow.org/get_started/get_started
# It was rewritten following PEP8 (except 79 char -> 119 char for PyCharm) with additional notes in between the snippets

# Tensor: a fundamental unit of data in TensorFlow, a set of primitive values in an array of any dimension
#         The rank of a tensor is its number of dimensions:
# 3 # a rank 0 tensor; this is a scalar with shape []
# [1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

# TensorFlow Core: the lowest level API for TensorFlow. Consists two components:
# 1. Buildling the computational graph
# 2. Running the computational graph


def main():
    print("TensorFlow Core tutorial")

    # Computational graph: series of TensorFlow operations arranged into a graph of nodes
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly
    # This should print out the following:
    # "Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)"
    print(node1, node2)

    # Note that the nodes' values are not printed. To evaluate the nodes, they must be run within a "session"
    sess = tf.Session()
    print(sess.run([node1, node2]))  # Output: "[3.0, 4.0]"

    # The nodes can be added using tf.add to build more complicated computation
    node3 = tf.add(node1, node2)
    print("node3: ", node3)
    print("sess.run(node3): ", sess.run(node3))

    # Instead of initializing nodes with value(s), a placeholder can be used to initialize without an initial value
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)
    print(sess.run(adder_node, {a: 3, b: 4.5}))  # Output: 7.5
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))  # Output: [3. 7.]

    # The computational graph can be further expanded by additional operations
    add_and_triple = adder_node * 3.
    print(sess.run(add_and_triple, {a: 3, b:4.5}))  # Output: 22.5

    # Variables can be created using tf.Variable with a type and initial value
    w = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = w * x + b

    # Note that tf.Variable doesn't initialize the variable
    # All variables must be initialized using the following lines:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))  # Output: [0. 0.30000001 0.60000002 0.90000004]

    # Evaluating the linear model on the training data x & y
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)  # Error of individual prediction
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))  # Output: 23.66 - that's pretty bad :(

    # Fixing the linear model with better values for each variable
    fix_w = tf.assign(w, [-1.])
    fix_b = tf.assign(b, [1.])
    sess.run([fix_w, fix_b])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))  # Output: 0
    return

if __name__ == '__main__':
    main()
