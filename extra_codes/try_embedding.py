## Will average the frame-level features for a video
## and represent in t-SNE
import __init__
base_path = __init__.base_path

import os, sys
import numpy as np
from glob import glob

from sklearn.manifold import TSNE


from utils import load_pkl, get_labels, get_ids, dump_pkl

from time import time

num_samples = 5043
featureName = 'vgg'
featuresPath = os.path.join(base_path, 'data','frameFeatures','{}Features'.format(featureName))


trainLabels, valLabels, testLabels = get_labels()
trainIds, valIds, testIds = get_ids()

print trainLabels.shape
print valLabels.shape
print testLabels.shape

print len(trainIds)
print len(valIds)
print len(testIds)


def avg_frames(ID):

    ID = str(ID)
    pklPath = os.path.join(featuresPath, ID+'.p')
    data = load_pkl(pklPath, verbose=True)
    avg = np.mean(data, axis=0)
    return avg

start = time()
_limit = None
valAvgs = np.array([avg_frames(ID) for ID in testIds[:_limit]])
print valAvgs

mode = 'test'
lr = [10,20,50,100,150,200,250,300,400,500]

for learning_rate in lr:
    X_embedded = TSNE(n_components=2, learning_rate=learning_rate, perplexity=50).fit_transform(valAvgs)
    dump_pkl(X_embedded, 'tsne_plots/tsne_2d_{}_{}_{}.p'.format(featureName, mode, learning_rate))
    print time() - start, "seconds."
###
