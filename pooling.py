## code for fastVideo - pooling of frames
## for genre classification

## to run, call python pooling.py <featureName> <pooling_operation>

import numpy as np
import os, sys
base_path = os.getcwd()
if not os.path.basename(base_path) == 'moviescope':
    base_path = os.path.join(base_path, '..')

from utils import load_pkl, get_labels, get_ids, dump_pkl
from evaluations import find_precision_recall

num_genres = 13

#featureName can be either 'vgg' or 'res'
featureName = sys.argv[1]

featuresPath = os.path.join(base_path, 'data','frameFeatures','{}Features'.format(featureName))

trainLabels, valLabels, testLabels = get_labels()
trainIds, valIds, testIds = get_ids()

def avg_frames(ID):

    ID = str(ID)
    pklPath = os.path.join(featuresPath, ID+'.p')
    data = load_pkl(pklPath, verbose=True)
    avg = np.mean(data, axis=0)
    return avg

def max_frames(ID):

    ID = str(ID)
    pklPath = os.path.join(featuresPath, ID+'.p')
    data = load_pkl(pklPath, verbose=True)
    avg = np.max(data, axis=0)
    return avg

   
## pool_operation can be 'max' or 'average'
pool_operation = sys.argv[2]

pool_func = {'max':max_frames, 'average':avg_frames}[pool_operation]


outPath = os.path.join(base_path,'data','inputFeatures',pool_operation)
if not os.path.exists(outPath):
    os.makedirs(outPath)
    
_write_data = not os.path.exists(os.path.join(outPath, '{}_train.p'.format(featureName)))
        
if _write_data:
    x_train = np.array([pool_func(ID) for ID in trainIds])
    x_val = np.array([pool_func(ID) for ID in valIds])
    x_test = np.array([pool_func(ID) for ID in testIds])

    dump_pkl(x_train, os.path.join(outPath,'{}_train.p'.format(featureName)))
    dump_pkl(x_val, os.path.join(outPath,'{}_val.p'.format(featureName)))
    dump_pkl(x_test, os.path.join(outPath,'{}_test.p'.format(featureName)))
else:
    x_train = load_pkl(os.path.join(outPath,'{}_train.p'.format(featureName)))
    x_val = load_pkl(os.path.join(outPath,'{}_val.p'.format(featureName)))
    x_test = load_pkl(os.path.join(outPath,'{}_test.p'.format(featureName)))

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD, Adam

avg_input = Input(shape=(x_train.shape[1],))
hidden = Dense(512, activation='relu')(avg_input)
hidden = Dense(64, activation='relu')(hidden)
prediction = Dense(num_genres, activation='sigmoid')(hidden)

model = Model(inputs=avg_input, outputs=prediction)

optim = Adam(lr=0.0001, decay=0.001)
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

epochs = 2000
batch_size = 32

best_ap = 0.0

#history = {'precision':[],'recall':[],'avg_pr':[],'train_loss':[],'val_loss':[]}
history = {'val_avg_pr':[],'train_avg_pr':[],'train_loss':[],'val_loss':[]}
with open('experiment_stats/'+pool_operation,'a') as statFile:
    statFile.write('\nExperiment '+pool_operation+'\n')

modelSavePath = os.path.join(base_path,'trained_models','video','model_'+pool_operation+'.h5')
for epoch in range(1, epochs+1):
    h = model.fit(x_train, trainLabels, epochs=1, batch_size=batch_size, validation_data=(x_val, valLabels))
    yPreds = model.predict(x_train)
    precision, recall, avg_pr = find_precision_recall(trainLabels, yPreds)
    #history['precision'].append(precision)
    #history['recall'].append(recall)
    history['train_avg_pr'].append(avg_pr)
    train_ap = avg_pr['micro']

    yPreds = model.predict(x_val)
    precision, recall, avg_pr = find_precision_recall(valLabels, yPreds)
    #history['precision'].append(precision)
    #history['recall'].append(recall)
    history['val_avg_pr'].append(avg_pr)

    history['train_loss'].append(h.history['loss'][0])
    history['val_loss'].append(h.history['val_loss'][0])

    val_ap = avg_pr['micro']

    if val_ap > best_ap:
        #model.save(modelSavePath)
        model.save_weights('weights_vis.h5')
        best_ap = val_ap

    print 'Epoch:{} Train AP:{} Val AP: {}'.format(epoch, train_ap, val_ap)
    with open('experiment_stats/'+pool_operation, 'a') as statFile:
        statFile.write('Epoch:{} Train AP:{} Val AP: {}\n'.format(epoch, train_ap, val_ap))

historyPath = os.path.join(base_path,'history_log','{}_{}.p'.format(featureName, pool_operation))
dump_pkl(history,historyPath)
