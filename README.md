# CTC + Tensorflow

## Install Requirements

`pip install -r requirements.txt`

- Python 2.7+
- Tensorflow 1.0+
- python_speech_features
- numpy
- scipy

## To train
1. Set hyperparameters in conf.json
```
"FEATURES": 13,
"CLASSES": 41,
"HIDDEN": 256,
"LAYERS": 1,
"BATCH_SIZE": 32,
"INITIAL_LEARNING_RATE": 0.001,
"MOMENTUM": 0.9
```
2. Run:
`python train.py`

## To freeze
after training for a few days, call freeze.py to create a frozen graph
`python freeze.py`

## To run inference
After running freeze.py, a frozen_graph file (.pb) will appear somewhere,
use label.py with

python label.py --audio_file_path="./timit_raw/DR1/FCJF0/SX307.WAV"

arguments:
--frozen_graph_path: path of .pb file outputted from freeze.py
--audio_file_path: location of audio file you want to run inference on

## TODO:
(try with more than 1 layer next time, perhaps 4?)
train on individual words (use master corpus?)
freeze.py
label.py
dropout
confusion matrix
