# will write all recurrent models
# like LSTM, BiLSTM, etc.

import __init__
from keras.layers import LSTM, Dense, Input, Bidirectional

def bidir_lstm_model(input_layer):
    lstm_1 = Bidirectional(LSTM(128, return_sequences=False, activation='sigmoid', recurrent_activation='tanh',
        name='bi_lstm'))(input_layer)
    hidden = Dense(64, activation='relu', name='hidden')(lstm_1)
    return hidden 

def stacked_bidir_model(input_layer):
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True, activation='sigmoid',
        recurrent_activation='tanh'))(input_layer)
    lstm_2 = Bidirectional(LSTM(128, return_sequences=False, activation='sigmoid', recurrent_activation='tanh',
        name='bi_lstm'))(lstm_1)
    hidden = Dense(64, activation='relu', name='hidden')(lstm_2)
    return hidden 


def lstm_model(input_layer):
    lstm_1 = LSTM(256, return_sequences=False, activation='sigmoid', recurrent_activation='sigmoid', name='lstm')(input_layer)
    hidden = Dense(64, activation='relu', name='hidden')(lstm_1)
    return hidden 

