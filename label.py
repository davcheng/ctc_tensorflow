import argparse
import sys
import numpy as np
import time

import scipy.io.wavfile as wav

import tensorflow as tf

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

FLAGS = None

# used to translate phones into indices
phone_index = {1: 'IY', 2: 'IH', 3: 'EH', 4: 'AE', 5: 'AH', 6: 'UW', 7: 'UH', 8: 'AA', 9: 'AO', 10: 'EY', 11: 'AY', 12: 'OY', 13: 'AW', 14: 'OW', 15: 'ER', 16: 'L', 17: 'R', 18: 'W', 19: 'Y', 20: 'M', 21: 'N', 22: 'NG', 23: 'V', 24: 'F', 25: 'DH', 26: 'TH', 27: 'Z', 28: 'S', 29: 'ZH', 30: 'SH', 31: 'JH', 32: 'CH', 33: 'B', 34: 'P', 35: 'D', 36: 'T', 37: 'G', 38: 'K', 39: 'HH',' ': ' '}
index_phone = {'AA': 8, 'W': 18, 'DH': 25, 'Y': 19, 'HH': 39, 'B': 33, 'JH': 31, 'ZH': 29, 'D': 35, 'NG': 22, 'TH': 26, 'IY': 1, 'CH': 32, 'AE': 4, 'EH': 3, 'G': 37, 'F': 24, 'AH': 5, 'K': 38, 'M': 20, 'L': 16, 'AO': 9, 'N': 21, 'IH': 2, 'S': 28, 'R': 17, 'EY': 10, 'T': 36, 'AW': 13, 'V': 23, 'AY': 11, 'Z': 27, 'ER': 15, 'P': 34, 'UW': 6, 'SH': 30, 'UH': 7, 'OY': 12, 'OW': 14}

# converts audio file into mfcc features to run inference with
def convert_wav_to_mfcc_inputs(audio_file_path):
    inference_fs, inference_audio = wav.read(audio_file_path)
    inference_inputs = mfcc(inference_audio, samplerate=inference_fs, numcep=13)
    # inputs = mfcc(audio, samplerate=fs, numcep=26) # make 26 features in filterbank
    # Tranform in 3D array
    inference_inputs = np.asarray(inference_inputs[np.newaxis, :])
    inference_inputs = (inference_inputs - np.mean(inference_inputs))/np.std(inference_inputs)
    inference_seq_len = [inference_inputs.shape[1]]
    return inference_inputs, inference_seq_len

# this loads a saved graph file (the variables) that was "frozen" with freeze.py
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) # Saved file must have as_text=False or else will get a "Error parsing message" error

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        tf.import_graph_def(graph_def, name="ctc") # this name will prefix each operation in your graph: '[PREFIX]/Placeholder/input_something_something'
    return graph

def main(_):
    print('Loading graph from: %s' % FLAGS.frozen_graph_path)
    # Load graph via "load_graph" function
    graph = load_graph(FLAGS.frozen_graph_path)

    # Uncomment this to show list of operations in the graph (helpful for debugging)
    # for op in graph.get_operations():
    #     print(op.name)
    # ctc/Placeholder/inputs_placeholder...

    # Gather MFCC data
    wav_input, seq_len_input = convert_wav_to_mfcc_inputs(FLAGS.audio_file_path)

    y = graph.get_tensor_by_name('ctc/CTCBeamSearchDecoder:0')

    # Launch a Session
    with tf.Session(graph=graph) as sess:

        start = time.time()

        feed_dict = {'ctc/input_data:0':wav_input,
                     'ctc/seq_len:0': seq_len_input}

        labels = sess.run(y, feed_dict=feed_dict)

        print('Time %s' % str(time.time()-start))

        str_coded = []
        str_decoded = []
        for x in np.asarray(labels[1]):
            if x > 0:
                str_coded.append(x)
            elif x == 0:
                str_coded.append(' ')

        for x in str_coded:
            str_decoded.append(phone_index[x])

        print('Decoded:\n%s' % str_decoded)

        str_coded_2 = []
        str_decoded_2 = []
        for x in np.asarray(labels[2]):
            if x > 0:
                str_coded_2.append(x)
            elif x == 0:
                str_coded_2.append(' ')

        for x in str_coded_2:
            str_decoded_2.append(phone_index[x])

        print('Decoded:\n%s' % str_decoded_2)

if __name__ == '__main__':
    # Allow user to pass in frozen filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen_graph_path",
        default="./frozen_graph_20180405.pb",
        type=str,
        help="Frozen model file to import")

    parser.add_argument(
        "--audio_file_path",
        default="./timit_raw/DR1/FCJF0/SI1027.WAV",
        type=str,
        help="Audio file path")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
