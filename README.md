# CTC + Tensorflow

## Install Requirements
Create virtualenv
```
virtualenv venv
source venv/bin/activate
```

Install packages into virtualenv
`pip install -r requirements.txt`

- Python 3.5+
- Tensorflow 1.4
- python_speech_features
- numpy
- scipy
- namedtupled (used for conf.json)

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

label.py takes two arguments:
- frozen_graph_path: path of .pb file outputted from freeze.py
- audio_file_path: location of audio file you want to run inference on

## To debug using tensorboard
Tensorboard comes installed with tensorflow; to make use of it, point the logdir flag to the location where the FileWriter serialized its data (currently set to `./tmp/`)
```
tensorboard --logdir=./log/ctc
```

## Training Notes
### CTC
- After 30 hours of training on the TIMIT and appspot word data, using AdamOptimizer, no gradient clipping,  dropout, and the following hyperparameters:
```
    "FEATURES": 13,
    "CLASSES": 41,
    "HIDDEN": 256,
    "LAYERS": 1,
    "BATCH_SIZE": 32,
    "INITIAL_LEARNING_RATE": 0.001,
    "MOMENTUM": 0.9
```
  Results are as follows:
```
  Epoch 111/200, train_cost = 27.383, train_ler = 0.204, val_cost = 38.645, val_ler = 0.229, time = 1026.488
  Original:
  even then, if she took one step forward he could catch her
  Decoded:
  ['IY', 'V', 'IH', 'N', ' ', 'DH', 'AE', 'N', ' ', 'IH', 'F', ' ', 'Y', 'IY', ' ', 'T', 'UW', ' ', 'W', 'AH', 'N', 'S', 'T', 'EH', 'P', ' ', 'F', 'AO', ' ', 'HH', 'IY', ' ', 'K', 'UH', ' ', 'K', 'AE', 'CH', ' ', 'ER']
```
  Note: Could sounds like "Cou" in the audio, which is positive since the detection only caught the "K UH"

- After 38 hours of training on the TIMIT and appspot word data, using AdamOptimizer, no gradient clipping,  dropout, and the same hyperparameters as above:
```
  Epoch 140/200, train_cost = 22.661, train_ler = 0.165, val_cost = 32.507, val_ler = 0.250, time = 1000.521
  saved to ctc_checkpoints/model for epoch: 139
  Original:
  even then, if she took one step forward he could catch her
  Decoded:
  ['IY', 'V', 'IH', 'N', ' ', 'EH', 'N', 'D', ' ', 'IH', 'F', ' ', 'IY', ' ', 'T', 'UW', 'K', ' ', 'K', 'W', 'AH', 'N', 'S', 'T', 'EH', 'P', 'F', 'AO', 'L', 'IH', ' ', 'HH', 'IY', ' ', 'K', 'UH', 'D', ' ', 'K', 'AE', 'CH', ' ', 'ER']
```
Note: seems to be overfit

### BDLSTM
- After 12 hours of training on the TIMIT and appspot word data, using AdamOptimizer, no gradient and 3 layers:
```
      "MODEL_ARCHITECTURE": "ctc",
      "FEATURES": 13,
      "CLASSES": 41,
      "HIDDEN": 256,
      "LAYERS": 3,
      "BATCH_SIZE": 32,
      "INITIAL_LEARNING_RATE": 0.001,
      "MOMENTUM": 0.9,
      "KEEP_PROB": 1.0
    ```
      Results are as follows:
    ```
  Epoch 27/200, train_cost = 33.505, train_ler = 0.239, val_cost = 56.929, val_ler = 0.375, time = 1806.252
  INFO:tensorflow:saved to ctc_checkpoints/model for epoch: 26
  Original:
  even then, if she took one step forward he could catch her
  Decoded:
  ['IY', ' ', 'UW', 'V', 'IH', 'N', ' ', 'EH', 'N', ' ', 'IH', ' ', 'IY', ' ', 'T', 'UW', 'W', 'AH', 'N', 'S', 'IH', 'P', ' ', 'F', 'AO', 'W', 'AH', ' ', 'HH', 'IY', 'K', 'UH', ' ', 'K', 'AE', 'CH', 'ER']
```

-  After 10 hours of training
```
    "MODEL_ARCHITECTURE": "bdlstm",
    "FEATURES": 13,
    "CLASSES": 41,
    "HIDDEN": 256,
    "LAYERS": 1,
    "BATCH_SIZE": 32,
    "INITIAL_LEARNING_RATE": 0.001,
    "MOMENTUM": 0.9,
    "KEEP_PROB": 1.0
```
Results:
```
Epoch 17/200, train_cost = 37.230, train_ler = 0.270, val_cost = 65.126, val_ler = 0.396, time = 2699.885
check graph exists
INFO:tensorflow:saved to bdlstm_checkpoints/model for epoch: 16
Original:
even then, if she took one step forward he could catch her
Decoded:
['IH', ' ', 'UW', 'D', 'IH', 'N', ' ', 'N', 'DH', 'EH', 'N', ' ', 'IH', 'SH', 'IY', 'T', 'UW', 'K', 'W', 'AH', 'N', 'S', 'T', 'EH', 'P', 'F', 'AO', ' ', 'W', 'AH', ' ', 'HH', 'IY', ' ', 'K', 'UH', 'K', 'AE', 'CH', 'ER']
```
```
Epoch 25/200, train_cost = 17.126, train_ler = 0.098, val_cost = 30.098, val_ler = 0.208, time = 2972.312
check graph exists
INFO:tensorflow:saved to bdlstm_checkpoints/model for epoch: 24
Original:
even then, if she took one step forward he could catch her
Decoded:
['IY', 'V', 'IH', 'N', ' ', 'DH', 'EH', 'N', ' ', 'IH', ' ', 'SH', 'IY', ' ', 'T', 'UH', 'K', ' ', 'W', 'AH', 'N', 'S', 'T', 'EH', 'T', ' ', 'AO', 'W', 'OW', ' ', 'HH', 'IY', ' ', 'K', 'UH', ' ', 'K', 'AE', 'CH', 'ER']
```
24 hours of training
```
Epoch 38/200, train_cost = 4.169, train_ler = 0.011, val_cost = 7.663, val_ler = 0.042, time = 2645.201
check graph exists
INFO:tensorflow:saved to bdlstm_checkpoints/model for epoch: 37
Original:
even then, if she took one step forward he could catch her
Decoded:
['IY', 'V', 'IH', 'N', ' ', 'DH', 'EH', 'N', ' ', 'IH', ' ', 'SH', 'IY', ' ', 'T', 'UH', ' ', 'K', ' ', 'W', 'AH', 'N', ' ', 'S', 'T', 'EH', 'P', ' ', 'F', 'AO', 'R', 'W', 'ER', 'D', ' ', 'HH', 'IY', ' ', 'K', 'UH', 'D', ' ', 'K', 'AE', 'CH', ' ', 'HH', 'ER']
```

BDLSTM, 3 layers, TIMIT + Appspot + speech commands 0.2 set
13 hrs ... (need GPUs or more powerful machine)
```
Epoch 2/200, train_cost = 11.396, train_ler = 0.609, val_cost = 148.533, val_ler = 0.833, time = 17400.770
check graph exists
INFO:tensorflow:saved to bdlstm_checkpoints/model for epoch: 1
Original:
even then, if she took one step forward he could catch her
Decoded:
['IY', ' ', 'AH', ' ', 'AH', ' ', 'L', ' ', 'AH', ' ', 'ER', 'N']

Epoch 3/200, train_cost = 7.095, train_ler = 0.288, val_cost = 129.235, val_ler = 0.729, time = 18041.436
check graph exists
INFO:tensorflow:saved to bdlstm_checkpoints/model for epoch: 2
Original:
even then, if she took one step forward he could catch her
Decoded:
[' ', 'IY', ' ', 'AH', ' ', 'AH', ' ', 'L', ' ', 'S', 'IH', ' ', 'F', 'L', ' ', 'IY', ' ', 'IH', ' ', 'AE', 'S']
```
after night 2...
```
Epoch 8/200, train_cost = 4.059, train_ler = 0.126, val_cost = 106.483, val_ler = 0.583, time = 13856.706
check graph exists
INFO:tensorflow:saved to bdlstm_checkpoints/model for epoch: 7
Original:
even then, if she took one step forward he could catch her
Decoded:
['IY', ' ', 'Y', 'IH', 'N', ' ', 'M', 'AE', 'N', ' ', 'IH', 'S', 'SH', 'AH', 'T', 'IH', ' ', 'P', 'W', 'AH', 'N', 'S', 'IH', 'F', 'AO', ' ', 'IY', 'K', 'AH', ' ', 'K', 'AE', 'N', 'S', 'AH']
```

## TODO:
(try with more than 1 layer next time, perhaps 4?)
train on individual words (use master corpus?)
- [X] Complete freeze.py
- [X] Complete  label.py
- [ ] Make train use batches for testing [see ex] (https://github.com/philipperemy/tensorflow-ctc-speech-recognition/blob/master/ctc_tensorflow_example.py)
- [X] Refactor out model into models.py
- [X] Accomodate more than 1 layer (SOLVED, for now: num_features must equal num_hidden_layers; otherwise only 1 layer works; [2*num_hidden] == [num_features+num_hidden]); https://github.com/tensorflow/tensorflow/issues/14897
- [X] Propogate models.py over to freeze
- [ ] Shuffle order of training/validation data
- [ ] Fix Single FC layer (maybe this isn't worth fixing...)
- [ ] Add dropout to improve model architecture
- [ ] Confusion matrix for phoneme performance evaluation
- [X] Train model with TIMIT
- [X] Train model with TIMIT and Appspot words
- [ ] Train model with TIMIT and Appspot words and speech commands
- [ ] Train model with TIMIT and Appspot words and verified extracted and master corpus
- [ ] GPU implementation [(see docs)](https://www.tensorflow.org/programmers_guide/using_gpu)
