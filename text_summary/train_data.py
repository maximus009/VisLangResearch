import __init__
base_path = __init__.base_path

import os
import numpy as np
from time import time

from utils import load_pkl
from evaluations import find_precision_recall

from keras.layers import LSTM, Bidirectional
from keras.layers import Embedding, Dense, Input, GlobalAveragePooling1D
from keras.optimizers import Adam

from keras.models import Model
import keras.backend as K

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


model_name = 'glove'
bestWeightsPath = 'best_weights_%s.h5' % model_name
weightsPath = 'weights_%s.h5' % model_name

best_ap = 0.689
best_epoch = 0
epochs = 100

learning_rate = 1e-5

statFileName = 'exp_text_%s' % model_name
with open(statFileName, 'a') as statFile:
    statFile.write("\n---Experiment %s---" % model_name)

load_glove = False
embeddingWeights = load_pkl('glove_weights.p') if load_glove else None

for epoch in range(1,epochs+1):
   yPred_train, yActual_train = None, None
   yPred_val, yActual_val = None, None
   start = time()
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
       if load_best and os.path.exists(bestWeightsPath):
           model.load_weights(bestWeightsPath)
       else:
           if os.path.exists(weightsPath):
              model.load_weights(weightsPath)
       optim = Adam(lr=learning_rate)
       model.compile(loss='binary_crossentropy', optimizer='adam')
       model.train_on_batch(xt, yt)
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
       model.save_weights(weightsPath)
       load_best = False

   precision, recall, meanap = find_precision_recall(yActual_train, yPred_train)
   train_ap = meanap['micro']

   precision, recall, meanap = find_precision_recall(yActual_val, yPred_val)
   val_ap = meanap['micro']

   if val_ap > best_ap:
       model.save_weights(bestWeightsPath)
       best_ap = val_ap
       best_epoch = epoch
   K.clear_session()
   epochTime = time() - start
   print "Epoch: {}, train_ap: {}, val_ap: {}, time_s: {}, best: {} @ {}".format(epoch, train_ap, val_ap, epochTime, best_ap, best_epoch)
   with open(statFileName, 'a') as statFile:
       statFile.write("\nEpoch: {}, train_ap: {}, val_ap: {}, time_s: {}, best: {} @ {}".format(epoch, train_ap, val_ap, epochTime, best_ap, best_epoch))
statFile.close()
