# CTC + Tensorflow

## Install Requirements

`pip install -r requirements.txt`

- Python 2.7+
- Tensorflow 1.0+
- python_speech_features
- numpy
- scipy

## To train
Set hyperparameters in conf.json
```
"FEATURES": 13,
"CLASSES": 41,
"HIDDEN": 256,
"LAYERS": 1,
"BATCH_SIZE": 32,
"INITIAL_LEARNING_RATE": 0.001,
"MOMENTUM": 0.9
```
Once hyperparameters are set, call:
`python train.py`

## To freeze
After training for a few days, call freeze.py to create a frozen graph
```
python freeze.py
```

## To run inference
After running `freeze.py`, a frozen_graph (.pb) file will be created.
Use label.py to run inference
```
python label.py --audio_file_path='./timit_raw/DR1/FCJF0/SX307.WAV' --frozen_graph_path='/PATH/TO/PB/FILE'
```

arguments:
- frozen_graph_path: path of .pb file outputted from freeze.py
- audio_file_path: location of audio file you want to run inference on


## To debug using tensorboard
Tensorboard comes installed with tensorflow; to make use of it, point the logdir flag to the location where the FileWriter serialized its data (currently set to `./tmp/`)
```
tensorboard --logdir=./log/ctc
```

## Training Notes
- After 30 hours of training on the TIMIT and appspot word data, using AdamOptimizer, no gradient clipping,  dropout, and the following hyperparameters:
    "FEATURES": 13,
    "CLASSES": 41,
    "HIDDEN": 256,
    "LAYERS": 1,
    "BATCH_SIZE": 32,
    "INITIAL_LEARNING_RATE": 0.001,
    "MOMENTUM": 0.9

  Results are as follows:
  Epoch 111/200, train_cost = 27.383, train_ler = 0.204, val_cost = 38.645, val_ler = 0.229, time = 1026.488
  Original:
  even then, if she took one step forward he could catch her
  Decoded:
  ['IY', 'V', 'IH', 'N', ' ', 'DH', 'AE', 'N', ' ', 'IH', 'F', ' ', 'Y', 'IY', ' ', 'T', 'UW', ' ', 'W', 'AH', 'N', 'S', 'T', 'EH', 'P', ' ', 'F', 'AO', ' ', 'HH', 'IY', ' ', 'K', 'UH', ' ', 'K', 'AE', 'CH', ' ', 'ER']
  Note: Could sounds like "Cou" in the audio, which is positive since the detection only caught the "K UH"

- After 38 hours of training on the TIMIT and appspot word data, using AdamOptimizer, no gradient clipping,  dropout, and the same hyperparameters as above:

  Epoch 140/200, train_cost = 22.661, train_ler = 0.165, val_cost = 32.507, val_ler = 0.250, time = 1000.521
  check graph
  True
  saved to ctc_checkpoints/model for epoch: 139
  Original:
  even then, if she took one step forward he could catch her
  Decoded:
  ['IY', 'V', 'IH', 'N', ' ', 'EH', 'N', 'D', ' ', 'IH', 'F', ' ', 'IY', ' ', 'T', 'UW', 'K', ' ', 'K', 'W', 'AH', 'N', 'S', 'T', 'EH', 'P', 'F', 'AO', 'L', 'IH', ' ', 'HH', 'IY', ' ', 'K', 'UH', 'D', ' ', 'K', 'AE', 'CH', ' ', 'ER']

Note: seems to be overfit




## TODO:
(try with more than 1 layer next time, perhaps 4?)
train on individual words (use master corpus?)
- [X] Complete freeze.py
- [X] Complete  label.py
- [ ] Make train use batches for testing [see ex] (https://github.com/philipperemy/tensorflow-ctc-speech-recognition/blob/master/ctc_tensorflow_example.py)
- [X] Refactor out model into models.py
- [X] Accomodate more than 1 layer (SOLVED, for now: num_features must equal num_hidden_layers; otherwise only 1 layer works; [2*num_hidden] == [num_features+num_hidden]); https://github.com/tensorflow/tensorflow/issues/14897
- [ ] Fix Single FC layer (maybe this isn't worth fixing...)
- [ ] Add dropout to improve model architecture
- [ ] Confusion matrix for phoneme performance evaluation
- [X] Train model with TIMIT
- [X] Train model with TIMIT and Appspot words
- [ ] Train model with TIMIT and Appspot words and verified extracted
- [ ] Train model with TIMIT and Appspot words and verified extracted and master corpus
- [ ] GPU implementation [(see docs)](https://www.tensorflow.org/programmers_guide/using_gpu)
