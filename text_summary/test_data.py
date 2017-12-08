import __init__

import os
import numpy as np

from utils import load_pkl

from keras.layers import LSTM, Bidirectional
from keras.layers import Embedding, Dense, Input, GlobalAveragePooling1D

from keras.models import Model

big_data = load_pkl('big_data_summary_batches.p')

def core_model(embed_seq):
    lstm_1 = LSTM(128, recurrent_activation='tanh', name='lstm')(embed_seq)
    hidden = Dense(64, activation='relu')(lstm_1)
    return hidden

def generate_batches_length():
    for l in sorted(big_data):
        yield l,big_data[l]

def generate_batches():
    for l, (dt, dv) in generate_batches_length():
        xt, yt = zip(*dt)
        xv, yv = zip(*dv)
        print 'Generating data for length',l,'\r',
        x_train = np.empty((len(xt), l))
        y_train = np.empty((len(xt), 13))
        for i,plot in enumerate(xt):
            selected_index = sorted(np.random.choice(len(plot), l, replace=False))
            x_train[i,:] = plot[selected_index]
            y_train[i, :] = yt[i]
            
        x_val = np.empty((len(xv), l))
        y_val = np.empty((len(xv), 13))
        for i,plot in enumerate(xv):
            selected_index = sorted(np.random.choice(len(plot), l, replace=False))
            x_val[i,:] = plot[selected_index]
            y_val[i, :] = yv[i]
        yield l, x_train, y_train, x_val, y_val
    print


model_name = ''
bestWeightsPath = 'best_weights_%s.h5' % model_name

load_glove = False
embeddingWeights = load_pkl('glove_weights.p') if load_glove else None

yPred_train, yActual_train = None, None
yPred_val, yActual_val = None, None
load_best = True
for l, xt, yt, xv, yv in generate_batches():
   input_layer = Input(shape=(l,), dtype='int32')
   if load_glove:
       embedding = Embedding(50001, 300, input_length=l, weights=[embeddingWeights])
       load_glove = False
   else:
       embedding = Embedding(50001, 300, input_length=l)
   embed_seq = embedding(input_layer)
   fastText = GlobalAveragePooling1D()(embed_seq)
   dense_1 = Dense(64, activation='relu')(fastText)
   prediction = Dense(13, activation='sigmoid')(dense_1)
   model = Model(inputs=input_layer, outputs=prediction)
   model.load_weights(bestWeightsPath)

   if yPred_train is None:
       yPred_train = model.predict(xt)
       yActual_train = yt
       yPred_val = model.predict(xv)
       yActual_val = yv
   else:
       yPred_train = np.concatenate((yPred_train, model.predict(xt)))
       yActual_train = np.concatenate((yActual_train, yt))
       yPred_val = np.concatenate((yPred_val, model.predict(xv)))
       yActual_val = np.concatenate((yActual_val, yv))


np.save('ground_truth_train', yActual_train)
np.save('ground_truth_val', yActual_val)
np.save('pred_train_%s' % model_name, yPred_train)
np.save('pred_val_%s' % model_name, yPred_val)
