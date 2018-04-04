# CTC + Tensorflow

## Install Requirements

`pip install -r requirements.txt`

- Python 2.7+
- Tensorflow 1.0+
- python_speech_features
- numpy
- scipy

## To train
`python train.py`

## To freeze
after training for a few days, call freeze.py to create a frozen graph
`python freeze.py`

I used these hyperparaemters
```
num_epochs = 100
num_hidden = 256
num_layers = 1
batch_size = 32
initial_learning_rate = .001
momentum = 0.9
```
(try with more than 1 layer next time, perhaps 4?)
