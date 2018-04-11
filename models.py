"""
Model definitions for simple speech recognition.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
from constants import c

# Hyperparameters
# moved to conf.json, served with constants.py
num_features = c.CTC.FEATURES
# 39 phones + space + blank label (needed for CTC) = 41 classes
num_classes = c.CTC.CLASSES
num_hidden = c.CTC.HIDDEN # 32 default
num_layers = c.CTC.LAYERS # only works with one... gets a dimension error that is a product of num hidden
batch_size = c.CTC.BATCH_SIZE
initial_learning_rate = c.CTC.INITIAL_LEARNING_RATE
momentum = c.CTC.MOMENTUM


def create_model(model_architecture, model_inputs, is_training):
    """Builds a model of the requested architecture compatible with the settings.

    Args:
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.

    Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
    Raises:
    Exception: If the architecture type isn't recognized.
    """
    if model_architecture == 'ctc':
        return create_ctc_model(model_inputs, is_training)
    elif model_architecture == 'bdlstm':
        return create_bdlstm_model(model_inputs, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "ctc", "lstm",...')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_ctc_model(model_inputs, is_training):
    """ Builds a CTC model"""
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        # Add dropout for W
        # keep_prob = tf.placeholder(tf.float32)
        # dropout_prob = tf.nn.dropout(W, keep_prob)

    inputs, targets, seq_len = model_inputs

    # Define the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    # The second output is the last state and we will not use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    # Inputs shape
    input_shape = tf.shape(inputs)
    # Get shape; max_timesteps not used but cool to know (i guess... probably remove later)
    batch_size, max_timesteps = input_shape[0], input_shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    # Zero initialization
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Add dropout for W
    # keep_prob = tf.placeholder(tf.float32)
    # dropout_prob = tf.nn.dropout(W, keep_prob)

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b
    # logits = tf.matmul(outputs, W_drop) + b # Use this instead if you want to use dropout

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_bdlstm_model(model_inputs, is_training):
    """ Builds a CTC model
    The model is created by bidirectional dynamic rnn with three stacked lstms in each direction.
    http://www.cs.toronto.edu/~graves/icml_2006.pdf
    https://github.com/tbornt/phoneme_ctc/blob/master/train.py
    """

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        # Add dropout for W
        # keep_prob = tf.placeholder(tf.float32)
        # dropout_prob = tf.nn.dropout(W, keep_prob)

    inputs, targets, seq_len = model_inputs

    # Weights & biases
    # W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], mean=0, stddev=0.1, dtype=tf.float32))
    # b = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Network
    forward_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)
    backward_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)

    stack_forward_cell = tf.nn.rnn_cell.MultiRNNCell([forward_cell] * num_layers,
                                                     state_is_tuple=True)
    stack_backward_cell = tf.nn.rnn_cell.MultiRNNCell([backward_cell] * num_layers,
                                                      state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(stack_forward_cell,
                                                 stack_backward_cell,
                                                 inputs,
                                                 sequence_length=seq_len,
                                                 time_major=False, # [batch_size, max_time, num_hidden]
                                                 dtype=tf.float32)

    # Inputs shape
    input_shape = tf.shape(inputs)
    # Get shape; max_timesteps not used but cool to know (i guess... probably remove later)
    batch_size, max_timesteps = input_shape[0], input_shape[1]

    """
    outputs_concate = tf.concat_v2(outputs, 2)
    outputs_concate = tf.reshape(outputs_concate, [-1, 2*num_hidden])
    # logits = tf.matmul(outputs_concate, weight_classes) + bias_classes
    """
    fw_output = tf.reshape(outputs[0], [-1, num_hidden])
    bw_output = tf.reshape(outputs[1], [-1, num_hidden])
    logits = tf.add(tf.add(tf.matmul(fw_output, W), tf.matmul(bw_output, W)), b)

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    if is_training:
        return logits, dropout_prob
    else:
        return logits
        
