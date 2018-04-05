import os, argparse

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from constants import c

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
# https://stackoverflow.com/questions/38641887/how-to-save-a-trained-tensorflow-model-for-later-use-for-application
# https://stackoverflow.com/questions/46577833/using-bi-lstm-ctc-tensorflow-model-in-android
# https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html

# promising
# https://github.com/chiachunfu/speech/blob/master/export_lstm_pb.py

FLAGS = None

dir = os.path.dirname(os.path.realpath(__file__))

## NEED TO BE THE SAME AS train.py
# Some configs
num_features = c.CTC.FEATURES
num_classes = c.CTC.CLASSES
num_hidden = c.CTC.HIDDEN # 32 default
num_layers = c.CTC.LAYERS # only works with one... gets a dimension error that is a product of num hidden
batch_size = c.CTC.BATCH_SIZE
initial_learning_rate = c.CTC.INITIAL_LEARNING_RATE
momentum = c.CTC.MOMENTUM

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

    # Define the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # The second output is the last state and we will not use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32, time_major=False)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    decoded, log_prob = ctc.ctc_beam_search_decoder(logits, seq_len)
    y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
    # # Create an output to use for inference.
    # tf.nn.softmax(logits, name='ctc')

def freeze_graph(model_dir, output_node_names):
    # check if flags exist
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # Retrieve checkpoint path
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # Step 1: Create Session
    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:

        # Step 2: create inference graph (create an output to use for inference)
        # the name is used for the graph_tuil.convert_variables_to_constants
        # tf.nn.softmax(logits, name='labels_softmax')
        create_inference_graph()
        tf.global_variables_initializer().run()

        # Step 3: load models fromk checkpoints
        # speech commands does it through the following:
        # saver = tf.train.Saver(tf.global_variables())
        # saver.restore(sess, start_checkpoint)
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            sess.graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",")
            # output_node_names.split(",") # The output node names are used to select the usefull nodes: ex. ['labels_softmax']
        )
        tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname(FLAGS.model_dir),
            os.path.basename(FLAGS.output_file),
            as_text=False)
        tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)


        # # Finally we serialize and dump the output graph to the filesystem
        # with tf.gfile.GFile(output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        # print("%d ops in the final graph." % len(output_graph_def.node))

    return frozen_graph_def


    ### BEGIN TF
    # We clear devices to allow TensorFlow to control on which device it will load operations
    # clear_devices = True
    #
    # # Create the model and load its weights.
    # sess = tf.InteractiveSession()
    #
    # create_inference_graph(FLAGS.sample_rate,FLAGS.clip_duration_ms, FLAGS.clip_stride_ms,
    #     FLAGS.window_size_ms, FLAGS.window_stride_ms,
    #     FLAGS.dct_coefficient_count, FLAGS.model_architecture)
    #
    # models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    #
    # # Turn all the variables into inline constants inside the graph and save it.
    # frozen_graph_def = graph_util.convert_variables_to_constants(
    #     sess, sess.graph_def, ['labels_softmax'])
    # tf.train.write_graph(
    #     frozen_graph_def,
    #     os.path.dirname(FLAGS.output_file),
    #     os.path.basename(FLAGS.output_file),
    #     as_text=False)
    # tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)

    #### END TF

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
        default="SparseToDense",
        help="The name of the output nodes, comma separated.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="frozen_graph_20180405.pb",
        help="The name of the output nodes, comma separated.")
    FLAGS, unparsed = parser.parse_known_args()

    freeze_graph(FLAGS.model_dir, FLAGS.output_node_names)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
