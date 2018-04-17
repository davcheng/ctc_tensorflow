from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from constants import c
import models

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
# https://stackoverflow.com/questions/38641887/how-to-save-a-trained-tensorflow-model-for-later-use-for-application
# https://stackoverflow.com/questions/46577833/using-bi-lstm-ctc-tensorflow-model-in-android

# promising
# https://github.com/chiachunfu/speech/blob/master/export_lstm_pb.py

FLAGS = None

dir = os.path.dirname(os.path.realpath(__file__))

num_features = c.HYPERPARAMETERS.FEATURES
num_classes = c.HYPERPARAMETERS.CLASSES
num_hidden = c.HYPERPARAMETERS.HIDDEN # 32 default
num_layers = c.HYPERPARAMETERS.LAYERS # only works with one... gets a dimension error that is a product of num hidden
batch_size = c.HYPERPARAMETERS.BATCH_SIZE
initial_learning_rate = c.HYPERPARAMETERS.INITIAL_LEARNING_RATE
momentum = c.HYPERPARAMETERS.MOMENTUM

# MODEL_ARCHITECTURE = 'bdlstm'
MODEL_ARCHITECTURE = c.HYPERPARAMETERS.MODEL_ARCHITECTURE

# Construct the graph; should be exported to models.py
def create_inference_graph():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features], name='input_data')
    # create SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32, name='target_data')
    # create 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")


    # Create the model
    # passing in model inputs
    model_inputs = inputs, targets, seq_len
    # Abstracted Model architecture to models.py
    logits = models.create_model(model_architecture=MODEL_ARCHITECTURE, model_inputs=model_inputs, is_training=False)

    # This is where the CTC magic happens!
    # second output is log_probability, which we don't need
    decoded, _ = ctc.ctc_beam_search_decoder(logits, seq_len)

    # Create sparse tensor (not sure why right now...)
    # y = decoded[0]
    y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def main(_):
    # check if flags exist
    if not tf.gfile.Exists(FLAGS.model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % FLAGS.model_dir)
    if not FLAGS.output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # Create tensorflow session to load and freeze the graph
    with tf.Session(graph=tf.Graph()) as sess:

        # Create inference graph (create an output to use for inference)
        # the name is used for the graph_util.convert_variables_to_constants
        # tf.nn.softmax(logits, name='labels_softmax')
        create_inference_graph()

        # Automatically retrieve checkpoint path from given directory
        # Note: Can abstract this out and specify checkpoint if needed
        print('Loading model dir from directory: %s' % str(FLAGS.model_dir))
        checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        print('Input Checkpoint: %s' % str(input_checkpoint))
        load_variables_from_checkpoint(sess, input_checkpoint)

        # Turn all the variables into inline constants inside the graph and save it.
        # Note: There are many nodes, but only the output node really matters (because thats what the inference goes through)
        print('output_nodes: %s' % str(FLAGS.output_node_names))
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            sess.graph_def, # The graph_def is used to retrieve the nodes
            FLAGS.output_node_names.split(",") # output_node_names.split(",") # The output node names are used to select the usefull nodes: ex. ['labels_softmax']
        )

        # Write the graph to the specified output_file path
        tf.train.write_graph(
            frozen_graph_def,
            # sess.graph_def,
            os.path.dirname(FLAGS.model_dir),
            os.path.basename(FLAGS.frozen_graph_path),
            as_text=False)

        # Log
        tf.logging.info('Saved frozen graph to %s', FLAGS.frozen_graph_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./ctc_checkpoints",
        help="Model folder to export")
    parser.add_argument(
        "--output_node_names",
        type=str,
        # default="CTCBeamSearchDecoder",
        default="SparseToDense",
        help="The name of the output nodes, comma separated.")
    parser.add_argument(
        "--frozen_graph_path",
        type=str,
        default="frozen_graph_20180405.pb",
        help="The name of the output nodes, comma separated.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
