import __init__
base_path = __init__.base_path

import gc
import os, sys
import numpy as np
from utils import load_pkl, dump_pkl
from evaluations import find_precision_recall

from time import time

from keras import backend as K
from keras.layers import LSTM, Dense, Input, Bidirectional
from keras.models import Model, Sequential
from keras.optimizers import Adam

featureName = 'res'
model_name = sys.argv[1]
num_classes = 13
featuresPath = os.path.join(base_path,'data','frameFeatures','{}Features'.format(featureName))

data = os.path.join(base_path,'data','frames_length','{}'.format(featureName))

data = load_pkl(data)

weightsFolder = os.path.join(base_path,'data','weights')
if not os.path.exists(weightsFolder):
    os.makedirs(weightsFolder)

weightsPath = os.path.join(weightsFolder,'weights_{}_{}.h5'.format(model_name,featureName))
bestWeightsPath = os.path.join(weightsFolder,'best_weights_{}_{}.h5'.format(model_name,featureName))

batchesPath = os.path.join(base_path,'data','batches',featureName)

history = {'precision':[],'recall':[],'avg_pr':[]}


### to prevent re-running loading again and again
def get_batches(lengths):

    ## not giving default argument

    ## using generator to avoid memory boom
    ## just noticed there are no movies in validation set
    ## with 250 frames
    ## putting exception
    for l in lengths:
        # batch (xt,yt,xv,yv)
        if l == 250:
            continue
        xt,yt,xv,yv = load_pkl(os.path.join(batchesPath,'batch_data_{}_{}.p'.format(featureName,l)), verbose=False)
        yield xt,yt,xv,yv 


"""
### erased portion from previous code

"""

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
    lstm_1 = LSTM(256, return_sequences=False, recurrent_activation='tanh', name='lstm')(input_layer)
    hidden = Dense(64, activation='relu', name='hidden')(lstm_1)
    return hidden 

learning_rate = 0.0001
decay_rate = 0.01
best_ap = 0.47
best_epoch = 0.0

epochs = 100
batch_size = 32

lengths=range(25,301,25)
train_lengths =  lengths #'all' #[25,50,75,100,125]
eval_lengths = lengths # 'all' #[25,50,75,225,275,300]

history = { 'train_ap': dict([(l,[]) for l in train_lengths]),
            'val_ap': dict([(l,[]) for l in eval_lengths]),
            'val_precision': [],
            'val_recall': [],
            'val_avg_pr': [],
            'train_precision': [],
            'train_recall': [],
            'train_avg_pr': []}

if model_name == 'bidir_lstm':
    get_model = bidir_lstm_model
elif model_name == 'stacked_bidir':
    get_model = stacked_bidir_model 
else:
    get_model = lstm_model

for epoch in range(1, epochs+1):
    load_best = True

    print 'Starting Epoch:',epoch
    ## actual training
    learning_rate *= 1-decay_rate
    epoch_start_time = time()
    for x_train,y_train,x_val,y_val in get_batches(train_lengths):
        if x_val.tolist() == [] or x_train.tolist() == []:
            ### signifies no video with that num of frames in the validation set
            continue

        l = x_val[0].shape[0]
        input_layer = Input(shape = x_val[0].shape, name="inp")
        core_model = get_model(input_layer)
        prediction = Dense(num_classes, activation='sigmoid', name='out')(core_model)
        model = Model(inputs=[input_layer],outputs=prediction)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate))

        if load_best:
            if os.path.exists(bestWeightsPath):
                model.load_weights(bestWeightsPath)
        else:
            if os.path.exists(weightsPath):
                model.load_weights(weightsPath)

        load_best = False
        #model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)
        model.train_on_batch(x_train, y_train)

        ###
        #training meanAP

        y_predictions_train = model.predict(x_train)
        y_predictions_val = model.predict(x_val)

        _,_,avg_pr_train = find_precision_recall(y_train, y_predictions_train)
        _,_,avg_pr_val = find_precision_recall(y_val, y_predictions_val)
        train_ap, val_ap = avg_pr_train['micro'],avg_pr_val['micro']
        print '---\t Batch of {} frames. Training meanAP: {} \t Validation meanAP: {}. Learning Rate: {}'.format(l,train_ap,val_ap, learning_rate)
        history['train_ap'][l].append(train_ap)
        history['val_ap'][l].append(val_ap)
        ###
        #print 'Saving weights to',weightsPath
        model.save_weights(weightsPath)
    K.clear_session()

    ## evaluation

    #y_predictions = np.ones((1,num_classes))
    #y_actual = np.ones((1,num_classes))
    y_predictions = None
    y_actual = None

    t_predictions = None
    t_actual = None

    for x_train,y_train,x_val,y_val in get_batches(eval_lengths):

        input_layer = Input(shape = x_val[0].shape, name="input_layer")
        core_model = get_model(input_layer)
        prediction = Dense(num_classes, activation='sigmoid', name='out')(core_model)
        model = Model(inputs=[input_layer],outputs=prediction)

        if os.path.exists(weightsPath):
            model.load_weights(weightsPath)

        if y_predictions is None:
            y_predictions = model.predict(x_val)
            y_actual = y_val

            t_predictions = model.predict(x_train)
            t_actual = y_train
        else:
            y_predictions = np.concatenate((y_predictions,  model.predict(x_val)))
            y_actual = np.concatenate((y_actual, y_val))

            t_predictions = np.concatenate((t_predictions,  model.predict(x_train)))
            t_actual = np.concatenate((t_actual, y_train))

    epoch_end_time = time()

    precision, recall, avg_pr = find_precision_recall(t_actual, t_predictions)
    history['train_precision'].append(precision)
    history['train_recall'].append(recall)
    history['train_avg_pr'].append(avg_pr)
    train_acc = avg_pr['micro']

    precision, recall, avg_pr = find_precision_recall(y_actual, y_predictions)
    history['val_precision'].append(precision)
    history['val_recall'].append(recall)
    history['val_avg_pr'].append(avg_pr)
    ##if avg_pr['micro'] > best_ap:
    if avg_pr['micro'] > best_ap:
        model.save_weights(bestWeightsPath)
        best_ap = avg_pr['micro']
        print 'Saving best weights at',bestWeightsPath
        best_epoch = epoch

    K.clear_session()
    print "_"*40,model_name
    print 'End of Epoch:{} Train AP: {}\tVal AP:{}\t{} seconds.\t Current best AP: {} at epoch {}'.format(epoch, train_acc,
            avg_pr['micro'],epoch_end_time-epoch_start_time, best_ap, best_epoch)
    print "_"*40
    gc.collect()

historyPath = os.path.join(base_path,'history_log','{}_{}.p'.format(featureName, model_name))
if os.path.exists(historyPath):
    num = str(np.random.choice(10000))
    historyPath = historyPath.split('.')[0]+'_'+num+'.p'
dump_pkl(history,historyPath)
