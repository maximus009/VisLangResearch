import __init__

base_path = __init__.base_path

import os, sys
from pandas import read_csv 
import numpy as np

from utils import get_ids, get_labels

metadata_path = os.path.join(base_path,'data','movie_metadata.csv')
trainIds, valIds, testIds = get_ids()
trainLabels, valLabels, testLabels = get_labels()

inFile = open(metadata_path)
data = read_csv(inFile)

attributes = [
        'actor_1_name',
        #'actor_2_name',
        #'actor_3_name'
        ]

print 'Collecting all actor names'
actor_names = []
for attr in attributes:
    vals = data[attr].unique()
    actor_names += vals.tolist()

from collections import Counter
actor_names = sorted(actor_names, key=Counter(actor_names).get, reverse=True)

popular_actors = actor_names[:100]

print 'Loading...'
data_dict = dict(zip(range(5043),data['actor_1_name'].values))


x_train = []
y_train = []

print 'Collecting Train data...'
for ID,label in zip(trainIds, trainLabels):
    actor_index = [actor_names.index(data_dict[ID])+1]
    x_train.append(actor_index)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val = []
y_val = []

print 'Collecting Test data...'
for ID,label in zip(trainIds, trainLabels):
    actor_index = [actor_names.index(data_dict[ID])+1]
    x_val.append(actor_index)
    y_val.append(label)

x_val = np.array(x_val)
y_val = np.array(y_val)


if False:

    from keras.layers import Embedding, Dense, Flatten
    from keras.models import Sequential
    from keras.optimizers import SGD, Adam
    print 'Building model...'

    model = Sequential([Embedding(input_dim=len(actor_names), output_dim=200, input_length=1),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(13, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=10.0, decay=1e-3))

    print 'Training...'
    model.fit(x_train, y_train, validation_data=(x_val,y_val),epochs=5000,batch_size=32)
    model.save('model.h5')

from keras.models import load_model
model = load_model('model.h5')
embedding = model.layers[0].get_weights()[0]
from pickle import dump
dump((actor_names, embedding), open('actors_embeddings.p','wb'))

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, learning_rate=100, perplexity=50).fit_transform(embedding)
dump((actor_names,X_embedded),open('actor_2d_plots.p','wb'))
