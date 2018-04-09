from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange as range

import os
import sys
import numpy as np
import re
import scipy.io.wavfile as wav
from phoneme_model import phoneme_dict

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

# loops through folder of wavs and .txt targets
def load_batch_data(input_path):
    import os
    import re
    batch_train_inputs = []
    batch_train_targets = []
    batch_train_seq_len = []
    input_extensions = ['.wav']
    for subdir, dirs, files in os.walk(input_path):
        for file in files:
            if not re.match('SA', file):
                ext = os.path.splitext(file)[-1].lower()
                filename = os.path.splitext(file)[-2].lower()
                folder_name = os.path.split(subdir)[-1]
                # folder_name = os.path.splitext(file)[-3].lower()
                # For each wav file in folder
                if ext in input_extensions:
                    train_fs, train_audio = wav.read(os.path.join(subdir, file))
                    train_inputs = mfcc(train_audio, samplerate=train_fs)
                    # inputs = mfcc(audio, samplerate=fs, numcep=26) # make 26 features in filterbank
                    # Tranform in 3D array
                    # train_inputs = np.asarray(train_inputs[np.newaxis, :]) # giving this new axis makes appended shape go from (16,) to (144,13)
                    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
                    # train_seq_len = [train_inputs.shape[1]] # great for one example, but seq_len needs to be normalized for batch

                    # get the array index representation of the sentence label
                    # Note: this requres all wav files to have either a .txt with a line for the transcription
                    # OR requires the folder name of the sound file to be the label
                    try:
                        if os.path.exists(os.path.join(subdir, filename+'.txt')):
                            train_targets = preprocess_file(os.path.join(subdir, filename+'.txt'))
                        else:
                            # NOTE: This requires all
                            print('label .txt not in same file - parsing the folder directory for target')
                            print(folder_name)
                            train_targets = preprocess_text(folder_name)
                            print(train_targets)
                    except:
                        print('issue with preprocessing ' + filename) # most likely because it cannot convert a phoneme
                        pass

                    # print(np.asarray([train_inputs for i in train_inputs for i in train_inputs]).shape)
                    if all(x is not None for x in (train_inputs, train_targets)):
                        try:
                            batch_train_inputs.append(train_inputs) # this works if the array is not transformed into a 3D earlier
                            batch_train_targets.append(train_targets) # this WORKS, results in batch_train_targets with shape (num_examples,)
                            # print(np.asarray(train_inputs).shape)
                        except:
                            print("ERROR WITH FILE " + file)

    # print('shape') # (413, num_examples) if using batch_train_inputs.append
    # print(np.asarray(batch_train_inputs).shape) # need this to be (num_examples,), eg. (16,)
    return np.asarray(batch_train_inputs), np.asarray(batch_train_targets)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

def load_batched_data(specPath, targetPath, batchSize):
    import os
    '''returns 3-element tuple: batched data (list), max # of time steps (int), and
       total number of samples (int)'''
    return data_lists_to_batches([np.load(os.path.join(specPath, fn)) for fn in os.listdir(specPath)],
                                 [np.load(os.path.join(targetPath, fn)) for fn in os.listdir(targetPath)],
                                 batchSize) + \
            (len(os.listdir(specPath)),)


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)



####################
## Preprocess data
def preprocess_file(target_filename):
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    phone_index = {1: 'IY', 2: 'IH', 3: 'EH', 4: 'AE', 5: 'AH', 6: 'UW', 7: 'UH', 8: 'AA', 9: 'AO', 10: 'EY', 11: 'AY', 12: 'OY', 13: 'AW', 14: 'OW', 15: 'ER', 16: 'L', 17: 'R', 18: 'W', 19: 'Y', 20: 'M', 21: 'N', 22: 'NG', 23: 'V', 24: 'F', 25: 'DH', 26: 'TH', 27: 'Z', 28: 'S', 29: 'ZH', 30: 'SH', 31: 'JH', 32: 'CH', 33: 'B', 34: 'P', 35: 'D', 36: 'T', 37: 'G', 38: 'K', 39: 'HH',' ': ' '}
    index_phone = {'AA': 8, 'W': 18, 'DH': 25, 'Y': 19, 'HH': 39, 'B': 33, 'JH': 31, 'ZH': 29, 'D': 35, 'NG': 22, 'TH': 26, 'IY': 1, 'CH': 32, 'AE': 4, 'EH': 3, 'G': 37, 'F': 24, 'AH': 5, 'K': 38, 'M': 20, 'L': 16, 'AO': 9, 'N': 21, 'IH': 2, 'S': 28, 'R': 17, 'EY': 10, 'T': 36, 'AW': 13, 'V': 23, 'AY': 11, 'Z': 27, 'ER': 15, 'P': 34, 'UW': 6, 'SH': 30, 'UH': 7, 'OY': 12, 'OW': 14}

    with open(target_filename, 'r') as f:

        # Only the last line is necessary
        # 0 46797 She had your dark suit in greasy wash water all year.
        line = f.readlines()[-1]

        # Get only the words between [a-z] and replace period for none
        # this strips the first two numbers and only gives the tokens in the sentence, resulting in:
        # she had your dark suit in greasy wash water all year
        original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
        # original = ' '.join(line.strip().lower().split(' ')[2:]).replace('?', '')
        # original = ' '.join(line.strip().lower().split(' ')[2:]).replace('!', '')
        # regex to strip punctuation except apostrophe
        original = re.sub(r"[^\w\s']",'',original)

        # this adds a space so that when split, a space is preserved
        # she  had  your  dark  suit  in  greasy  wash  water  all  year
        targets = original.replace(' ', '  ')

        # split into array of words
        # ['she', '', 'had', '', 'your', '', 'dark', '', 'suit', '', 'in', '', 'greasy', '', 'wash', '', 'water', '', 'all', '', 'year']
        targets = targets.split(' ')

        # next convert array of words and spaces into phones and spaces
        # ['SH', 'IY', ' ', 'HH', 'AE', 'D', ' ', 'Y', 'AO', 'R', ' ', 'D', 'AA', 'R', 'K', ' ', 'S', 'UW', 'T', ' ', 'IH', 'N', ' ', 'G', 'R', 'IY', 'S', 'IY', ' ', 'W', 'AA', 'SH', ' ', 'W', 'AO', 'T', 'ER', ' ', 'AO', 'L', ' ', 'Y', 'IH', 'R']
        new_targets = []
        for word in targets:
            if len(word)>0:
                try:
                    new_targets.append(phoneme_dict[word.lower()])
                except:
                    print(word.lower() + ' not in phoneme_dict')
                    # pass
            else:
                new_targets.append(" ")
        targets = [phone for new_targets in new_targets for phone in new_targets]
        # print(targets)

    # Adding blank label (<space> instead of " "), removing commas
    # ['SH' 'IY' '<space>' 'HH' 'AE' 'D' '<space>' 'Y' 'AO' 'R' '<space>' 'D'
    #  'AA' 'R' 'K' '<space>' 'S' 'UW' 'T' '<space>' 'IH' 'N' '<space>' 'G' 'R'
    #  'IY' 'S' 'IY' '<space>' 'W' 'AA' 'SH' '<space>' 'W' 'AO' 'T' 'ER'
    #  '<space>' 'AO' 'L' '<space>' 'Y' 'IH' 'R']
    targets = np.hstack([SPACE_TOKEN if x == ' ' else x for x in targets])

    # Transform phoneme into index
    # [30  1  0 39  4 35  0 19  9 17  0 35  8 17 38  0 28  6 36  0  2 21  0 37
    # 17  1 28  1  0 18  8 30  0 18  9 36 15  0  9 16  0 19  2 17]
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else index_phone[x] for x in targets])
    return targets

# same as preprocess file, but takes a string input instead of a file with the label targets
def preprocess_text(text):
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    phone_index = {1: 'IY', 2: 'IH', 3: 'EH', 4: 'AE', 5: 'AH', 6: 'UW', 7: 'UH', 8: 'AA', 9: 'AO', 10: 'EY', 11: 'AY', 12: 'OY', 13: 'AW', 14: 'OW', 15: 'ER', 16: 'L', 17: 'R', 18: 'W', 19: 'Y', 20: 'M', 21: 'N', 22: 'NG', 23: 'V', 24: 'F', 25: 'DH', 26: 'TH', 27: 'Z', 28: 'S', 29: 'ZH', 30: 'SH', 31: 'JH', 32: 'CH', 33: 'B', 34: 'P', 35: 'D', 36: 'T', 37: 'G', 38: 'K', 39: 'HH',' ': ' '}
    index_phone = {'AA': 8, 'W': 18, 'DH': 25, 'Y': 19, 'HH': 39, 'B': 33, 'JH': 31, 'ZH': 29, 'D': 35, 'NG': 22, 'TH': 26, 'IY': 1, 'CH': 32, 'AE': 4, 'EH': 3, 'G': 37, 'F': 24, 'AH': 5, 'K': 38, 'M': 20, 'L': 16, 'AO': 9, 'N': 21, 'IH': 2, 'S': 28, 'R': 17, 'EY': 10, 'T': 36, 'AW': 13, 'V': 23, 'AY': 11, 'Z': 27, 'ER': 15, 'P': 34, 'UW': 6, 'SH': 30, 'UH': 7, 'OY': 12, 'OW': 14}

    line = text

    original = ' '.join(line.strip().lower().split(' ')).replace('.', '')
    # original = ' '.join(line.strip().lower().split(' ')[2:]).replace('?', '')
    # original = ' '.join(line.strip().lower().split(' ')[2:]).replace('!', '')
    # regex to strip punctuation except apostrophe
    original = re.sub(r"[^\w\s']",'',original)

    # this adds a space so that when split, a space is preserved
    # she  had  your  dark  suit  in  greasy  wash  water  all  year
    targets = original.replace(' ', '  ')

    # split into array of words
    # ['she', '', 'had', '', 'your', '', 'dark', '', 'suit', '', 'in', '', 'greasy', '', 'wash', '', 'water', '', 'all', '', 'year']
    targets = targets.split(' ')

    # next convert array of words and spaces into phones and spaces
    # ['SH', 'IY', ' ', 'HH', 'AE', 'D', ' ', 'Y', 'AO', 'R', ' ', 'D', 'AA', 'R', 'K', ' ', 'S', 'UW', 'T', ' ', 'IH', 'N', ' ', 'G', 'R', 'IY', 'S', 'IY', ' ', 'W', 'AA', 'SH', ' ', 'W', 'AO', 'T', 'ER', ' ', 'AO', 'L', ' ', 'Y', 'IH', 'R']
    new_targets = []
    for word in targets:
        if len(word)>0:
            try:
                new_targets.append(phoneme_dict[word.lower()])
            except:
                print(word.lower() + ' not in phoneme_dict')
                # pass
        else:
            new_targets.append(" ")
    targets = [phone for new_targets in new_targets for phone in new_targets]
    # print(targets)

    # Adding blank label (<space> instead of " "), removing commas
    # ['SH' 'IY' '<space>' 'HH' 'AE' 'D' '<space>' 'Y' 'AO' 'R' '<space>' 'D'
    #  'AA' 'R' 'K' '<space>' 'S' 'UW' 'T' '<space>' 'IH' 'N' '<space>' 'G' 'R'
    #  'IY' 'S' 'IY' '<space>' 'W' 'AA' 'SH' '<space>' 'W' 'AO' 'T' 'ER'
    #  '<space>' 'AO' 'L' '<space>' 'Y' 'IH' 'R']
    targets = np.hstack([SPACE_TOKEN if x == ' ' else x for x in targets])

    # Transform phoneme into index
    # [30  1  0 39  4 35  0 19  9 17  0 35  8 17 38  0 28  6 36  0  2 21  0 37
    # 17  1 28  1  0 18  8 30  0 18  9 36 15  0  9 16  0 19  2 17]
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else index_phone[x] for x in targets])
    return targets
