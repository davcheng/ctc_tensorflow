#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import date


import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
import re

from six.moves import xrange as range

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

from tensorflow.python.ops import ctc_ops as ctc

from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
from utils import preprocess_file as preprocess_file
from utils import load_batch_data as load_batch_data


from phoneme_model import phoneme_dict
from constants import c
import models


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

# used to translate phones into indices
phone_index = {1: 'IY', 2: 'IH', 3: 'EH', 4: 'AE', 5: 'AH', 6: 'UW', 7: 'UH', 8: 'AA', 9: 'AO', 10: 'EY', 11: 'AY', 12: 'OY', 13: 'AW', 14: 'OW', 15: 'ER', 16: 'L', 17: 'R', 18: 'W', 19: 'Y', 20: 'M', 21: 'N', 22: 'NG', 23: 'V', 24: 'F', 25: 'DH', 26: 'TH', 27: 'Z', 28: 'S', 29: 'ZH', 30: 'SH', 31: 'JH', 32: 'CH', 33: 'B', 34: 'P', 35: 'D', 36: 'T', 37: 'G', 38: 'K', 39: 'HH',' ': ' '}

# Training parameters
num_epochs = 200
num_examples = 32 # need to change this...
num_batches_per_epoch = int(num_examples/batch_size)

# needs to be a directory of folders with inputs (.wav) and labels (.txt), with corresponding labels in the same folder as the inputs
# note, remember to strip excess headers from TIMIT wavs using directory_audio_converter.sh
TRAIN_FILES_DIR = './timit_raw'
TEST_FILES_DIR = './timit_raw'

# MODEL_ARCHITECTURE = 'bdlstm'
MODEL_ARCHITECTURE = 'ctc'

#########
# PREPARE TEST DATA
print('loading test data')
test_audio_filename = './timit_raw/DR1/FCJF0/SI1027.WAV'
test_target_filename = './timit_raw/DR1/FCJF0/SI1027.TXT'

test_fs, test_audio = wav.read(test_audio_filename)
test_inputs = mfcc(test_audio, samplerate=test_fs)
# inputs = mfcc(audio, samplerate=fs, numcep=26) # make 26 features in filterbank
# Tranform in 3D array
test_inputs = np.asarray(test_inputs[np.newaxis, :])
batch_test_inputs = (test_inputs - np.mean(test_inputs))/np.std(test_inputs)
batch_test_seq_len = [test_inputs.shape[1]]
test_targets = preprocess_file(test_target_filename)
batch_test_targets = sparse_tuple_from([test_targets])
# end test data
######################


# main code
def main(_):
    # enable logging
    tf.logging.set_verbosity(tf.logging.INFO)
    print('Defining graph')
    graph = tf.Graph()
    with graph.as_default():

        # BUILD MODEL
        # TODO: replace with create_model() using models.py
        ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow


        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')
        # create SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32, name='target_data')
        # PERHAPS KEY TO NUMBER OF LAYERS ERROR?????
        # targets_idx = tf.placeholder(tf.int64)
        # targets_val = tf.placeholder(tf.int32)
        # targets_shape = tf.placeholder(tf.int64)
        # targets = tf.SparseTensor(targets_idx, targets_val, targets_shape)
        # create 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name="seq_len")

        # Create the model
        # passing in model inputs
        model_inputs = inputs, targets, seq_len
        # Abstracted Model architecture to models.py
        logits, dropout_prob = models.create_model(model_architecture=MODEL_ARCHITECTURE, model_inputs=model_inputs, is_training=True)

        # Define loss and optimizer
        # ctc cost function
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)
        # # Gradient clipping
        # tvars = tf.trainable_variables()
        # grads = tf.gradients(cost, tvars)
        # grad_norm = tf.global_norm(grads, name='grads')
        # grads, _ = tf.clip_by_global_norm(grads, 2, use_norm=grad_norm)
        # grads = list(zip(grads, tvars))
        # # Adam optimizer converges on solution faster than Momentum but is slower to train for each step
        optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate).minimize(cost)
        # optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)

        # perhaps test out?
        # optimizer = optimizer.apply_gradients(grads_and_vars=grads)

        # Option 1: tf.nn.ctc_greedy_decoder (faster, worse results)
        # Extra Knowledge: greedy is a special version of beam search where: top_paths=1 and beam_width=1
        # Option 2: tf.nn.ctc_beam_search_decoder (slower, better results)
        decoded, log_prob = ctc.ctc_beam_search_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))


        # used for tensorboard
        # Create a summary to monitor cost tensor
        summary_loss = tf.summary.scalar("loss", cost)
        # Create a summary to monitor accuracy tensor
        summary_ler = tf.summary.scalar("label_error_rate", ler)

        # Create summaries to visualize weights
        for var in tf.trainable_variables():
           tf.summary.histogram(var.name, var)

        # Summarize all gradients
        # summary_grad = tf.summary.scalar("gradient", grad_norm)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # Confusion Matrix
        # confusion_matrix = tf.confusion_matrix(targets, predicted_indices, num_classes=label_count)

    with tf.Session(graph=graph) as session:
        print("lets get this session going")
        try:
            merged = tf.summary.merge_all()
        except:
            merged = tf.summary.merge_all()
        try:
            writer = tf.summary.FileWriter("./log/ctc_20180406", session.graph)
        except:
            writer = tf.summary.FileWriter("./log/ctc_20180406", session.graph)
        try:
            saver = tf.train.Saver()  # defaults to saving all variables
        except:
            print("tf.train.Saver() broken in tensorflow 0.12")
            saver = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state('./ctc_checkpoints')

        print('done with try except block')
        start = 0
        if ckpt and ckpt.model_checkpoint_path:
            print('begin ckpt')
            p = re.compile('\./ctc_checkpoints/model\.ckpt-([0-9]+)')
            m = p.match(ckpt.model_checkpoint_path)
            try:
                start = int(m.group(1))
            except:
                pass
        if saver and start > 0:
            # Restore variables from disk.
            saver.restore(session, "./ctc_checkpoints/model.ckpt-%d" % start)
            print("Model %d restored." % start)
        else:
            # Initialize the weights and biases
            print('Initializing')
            try:
                session.run(tf.global_variables_initializer())
            except:
                session.run(tf.global_variables_initializer())

        # Load all of the training data
        print('Loading batch data from directory: %s' % TRAIN_FILES_DIR)
        train_inputs, train_targets = load_batch_data(TRAIN_FILES_DIR)
        print('Finished loading batch data')
        num_examples = train_targets.shape[0]
        print('Loaded %s samples' % str(num_examples))
        num_batches_per_epoch = int(num_examples/batch_size)

        # Begin training loop, resume from start if loaded previous checkpoints
        print('Begin training loop')
        tf.logging.info('Training from step: %d ' % start)
        for curr_epoch in range(start, num_epochs):
            # initialize training cost and label error rate
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):

                # indexes represent indices chosen per batch; if batch size=3, [0,1,2], [3,4,5]
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                # Create small batches out of the training data
                batch_train_inputs = train_inputs[indexes]

                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)

                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = sparse_tuple_from(train_targets[indexes])

                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*batch_size

            # Shuffle the data
            shuffled_indexes = np.random.permutation(num_examples)
            train_inputs = train_inputs[shuffled_indexes]
            train_targets = train_targets[shuffled_indexes]

            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples

            # Load validation data (in batches)
            # test_inputs, test_targets = load_batch_data(TEST_FILES_DIR)
            # # Create small batches out of the training data
            # batch_test_inputs = test_inputs[indexes]
            # # Padding input to max_time_step of this batch
            # batch_test_inputs, batch_test_seq_len = pad_sequences(batch_test_inputs)
            # # Converting to sparse representation so as to to feed SparseTensor input
            # batch_test_targets = sparse_tuple_from(test_targets[indexes])

            # validation data
            val_feed = {inputs: batch_test_inputs,
                        targets: batch_test_targets,
                        seq_len: batch_test_seq_len}

            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                             val_cost, val_ler, time.time() - start))

            # check to make sure using default graph so that saver works
            print('check graph')
            print(logits.graph == tf.get_default_graph())
            if saver:
                # saver.save(session, 'ctc_checkpoints/%s/model.ckpt', global_step=curr_epoch + 1 % str(date.today()))
                saver.save(session, 'ctc_checkpoints/model.ckpt', global_step=curr_epoch + 1)
                tf.logging.info('saved to ctc_checkpoints/model for epoch: %i' % curr_epoch)

            # Decoding
            # decoded is a tf variable
            d = session.run(decoded[0], feed_dict=val_feed)

            str_coded = []
            str_decoded = []
            for x in np.asarray(d[1]):
                if x > 0:
                    str_coded.append(x)
                elif x == 0:
                    str_coded.append(' ')

            for x in str_coded:
                str_decoded.append(phone_index[x])

            # Run inference on one file
            with open(test_target_filename, 'r') as f:

                # Only the last line is necessary
                # 0 46797 She had your dark suit in greasy wash water all year.
                line = f.readlines()[-1]

                # Get only the words between [a-z] and replace period for none
                # this strips the first two numbers and only gives the tokens in the sentence, resulting in:
                # she had your dark suit in greasy wash water all year
                original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')

            print('Original:\n%s' % original)
            print('Decoded:\n%s' % str_decoded)

        # # Turn all the variables into inline constants inside the graph and save it.
        # frozen_graph_def = graph_util.convert_variables_to_constants(session, session.graph_def, ['labels_softmax'])
        # output_file = './20180318_graph.pb'
        #
        # tf.train.write_graph(frozen_graph_def, os.path.dirname(output_file), os.path.basename(output_file), as_text=False)
        # # tf.train.write_graph(frozen_graph_def, './', '20180318_graph.pb', as_text=False)
        # tf.logging.info('Saved frozen graph to %s', output_file)

if __name__ == '__main__':
    tf.app.run(main=main)
