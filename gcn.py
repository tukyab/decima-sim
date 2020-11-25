"""
Graph Convolutional Network

Propergate node features among neighbors
via parameterized message passing scheme
"""

import copy
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tf_op import glorot, ones, zeros

class GraphCNN(object):
    def __init__(self, node_inputs, node_input_dim, edge_inputs, edge_input_dim,
                 hid_dims, output_dim, max_depth, act_fn, scope='gcn'):

        self.node_inputs = node_inputs
        self.edge_inputs = edge_inputs

        self.node_input_dim = node_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim

        self.max_depth = max_depth

        self.act_fn = act_fn
        self.scope = scope

        # message passing
        self.adj_mats = [tf.sparse_placeholder(
            tf.float32, [None, None]) for _ in range(self.max_depth)]
        self.masks = [tf.placeholder(
            tf.float32, [None, 1]) for _ in range(self.max_depth)]

        # W
        self.weights = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)

        self.prep_weights = glorot([self.node_input_dim, self.output_dim], scope=self.scope)

        # graph message passing
        self.outputs = self.forward()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))

        return weights

    def pan_filter(self):
        return 1

    def forward(self):
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.node_inputs
        h = self.edge_inputs

        x = tf.matmul(x, self.prep_weights)

        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x

            # process the features on the nodes
            for l in range(len(self.weights)):
                y = tf.matmul(y, self.weights[l])

                adj_tmp = tf.sparse.to_dense(self.adj_mats[d])
                adj = tf.sparse.to_dense(self.adj_mats[d])
                pan_adj = tf.sparse.to_dense(self.adj_mats[d])*self.pan_filter()
                for i in range(10):
                    adj_tmp = tf.matmul(adj_tmp, adj)
                    pan_adj = pan_adj + self.pan_filter() * adj_tmp

                rowsum = tf.math.reduce_sum(adj_tmp, 1)
                d_inv_sqrt = tf.math.pow(rowsum, -0.5)
                d_inv_sqrt = tf.reshape(d_inv_sqrt, [-1])
                d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
                norm = tf.tensordot(adj_tmp, d_mat_inv_sqrt, 1)
                norm = tf.transpose(norm)
                norm = tf.tensordot(norm, d_mat_inv_sqrt, 1)

                y = tf.multiply(norm, y)

            # message passing
            y = tf.sparse_tensor_dense_matmul(self.adj_mats[d], y)

            # assemble neighboring information
            x = y

        return x
