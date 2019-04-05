filename = 'text_scores.csv'

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
import numpy as np


def load_df(filename):
    raw = pd.read_csv(filename, index_col='index')
    df = pd.DataFrame(raw)
    X_train, y_train = df['text'], df['score']
    training = [(x, y) for x in X_train for y in y_train]
    return training


# load the text and score as a list of tuples
training = load_df(filename)

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

from keras.preprocessing import sequence

# create our training data from the text
textList = [x[0] for x in training]

# prepare the dataset of input
seqLen = 100
trainX = []
for text in textList:
    _x = [char_to_int[char] for char in text]
    assert len(_x) == 100
    trainX.append(_x)

n_patterns = len(trainX)
print("Total patterns : {}".format(len(trainX)))  # sanity check

trainX = np.divide(trainX, n_vocab)
trainY = np.asarray([x[1] for x in training])
trainX = np.reshape(trainX, (n_patterns, seqLen,))
print('shape: ', trainX.shape)
print('dasd', trainX[0])

model = Sequential()
# model.add(Embedding((n_patterns, seqLen,100),128))
model.add(LSTM(512, input_shape=(1, trainX.shape), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath = "weights-classifier-{epoch:02d}-{accuracy:0.4f}.hdf5"
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(trainX, trainY, epochs=3, batch_size=128, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(trainX, trainY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
