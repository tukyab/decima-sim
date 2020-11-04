"""
Graph Convolutional Network

Propergate node features among neighbors
via parameterized message passing scheme
"""

import copy
import numpy as np
import tensorflow as tf
from tf_op import glorot, ones, zeros

class GraphCNN(object):
    def __init__(self, node_inputs, node_input_dim, edge_inputs, edge_input_dim,
                 hid_dims, output_dim, max_depth, act_fn, scope='gcn'):

        self.node_inputs = node_inputs
        self.edge_inputs = edge_inputs

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
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

        # initialize message passing transformation parameters
        # h: x -> x'
        self.prep_weights, self.prep_bias = \
            self.init(self.node_input_dim, self.hid_dims, self.output_dim)

        self.edge_prep_weights, self.edge_prep_bias = \
            self.init(self.edge_input_dim, self.hid_dims, self.output_dim)

        # f: x' -> e
        self.proc_weights, self.proc_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)

        self.edge_proc_weights, self.edge_proc_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)

        # g: e -> e
        self.agg_weights, self.agg_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)

        # graph message passing
        self.outputs = self.forward()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def forward(self):
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.node_inputs
        h = self.edge_inputs

        # raise x into higher dimension
        for l in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[l])
            x += self.prep_bias[l]
            x = self.act_fn(x)

        for l in range(len(self.edge_prep_weights)):
            init = tf.placeholder(tf.float32, [h.shape.as_list()[-2], self.edge_prep_weights[l].shape.as_list()[-1]])
            # init = tf.reshape(h, [h.shape.as_list()[0], h.shape.as_list()[1], self.edge_prep_weights[l].shape.as_list()[-1]])
            h = tf.scan(lambda a, x: tf.matmul(x, self.edge_prep_weights[l]), h, init, infer_shape=False)
            h = tf.scan(lambda a, x: x + self.edge_prep_bias[l], h)
            h = tf.scan(lambda a, x: self.act_fn(x), h)

            # h = tf.matmul(h, self.edge_prep_weights[l])
            # h += self.edge_prep_bias[l]
            # h = self.act_fn(h)

            # if h.shape[0] != None:
            # for i in range(h.shape[0]):
            #     h[i] = tf.matmul(h[i], self.edge_prep_weights[l])
            #     h[i] += self.edge_prep_bias[l]
            #     h[i] = self.act_fn(h[i])

        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x
            e = h

            # process the features on the nodes
            for l in range(len(self.proc_weights)):
                y = tf.matmul(y, self.proc_weights[l])
                y += self.proc_bias[l]
                y = self.act_fn(y)

            # message passing
            y = tf.sparse_tensor_dense_matmul(self.adj_mats[d], y)

            # edge features
            for l in range(len(self.edge_proc_weights)):
                init = tf.placeholder(tf.float32, [h.shape.as_list()[-2], self.edge_proc_weights[l].shape.as_list()[-1]])
                # init = tf.reshape(h, [h.shape.as_list()[0], h.shape.as_list()[1], self.edge_proc_weights[l].shape.as_list()[-1]])
                h = tf.scan(lambda a, x: tf.matmul(x, self.edge_proc_weights[l]), h, init, infer_shape=False)
                h = tf.scan(lambda a, x: x + self.edge_proc_bias[l], h)
                h = tf.scan(lambda a, x: self.act_fn(x), h)

                # h = tf.matmul(h, self.edge_proc_weights[l])
                # h += self.edge_proc_bias[l]
                # h = self.act_fn(h)

                # if e.shape[0] != None:
                # for i in range(e.shape[0]):
                #     e[i] = tf.matmul(e[i], self.edge_proc_weights[l])
                #     e[i] += self.edge_proc_bias[l]
                #     e[i] = self.act_fn(e[i])

            # if y.shape[0] != None:
            #     for i in range(y.shape[0]):
            #         if e.shape[0] != None and e.shape[1] != None:
            #             a = e[i, 0]
            #             for j in range(1, e.shape[1]):
            #                     a += e[i, j]
            #             y[i] += a

            h = tf.reduce_sum(h, 1)
            y = tf.math.add(y, h)

            # aggregate child features
            for l in range(len(self.agg_weights)):
                y = tf.matmul(y, self.agg_weights[l])
                y += self.agg_bias[l]
                y = self.act_fn(y)

            # remove the artifact from the bias term in g
            y = y * self.masks[d]

            # assemble neighboring information
            x = x + y

        return x
