import os
from time import time

from evaluations import infer_flownet, infer_fastVideo, find_precision_recall
from utils import get_ids, get_labels

trainIds, valIds, testIds = get_ids()
trainLabels, valLabels, testLabels = get_labels()

base_path = os.getcwd()
if not os.path.basename(base_path) == 'moviescope':
    base_path = os.path.join(base_path, '..')

def test_fastVideo_single(_id=999):
    modelSavePath = os.path.join(base_path,'trained_models','video','model_average.h5')
    return infer_fastVideo(_id, model=modelSavePath)

def test_fastVideo_set(ids, labels, limit=None):
    modelSavePath = os.path.join(base_path,'trained_models','video','model_average.h5')
    s = time()
    preds = infer_fastVideo(ids[:limit], model=modelSavePath)
    print time()-s,'seconds'
    p, r, a = find_precision_recall(labels[:limit], preds)
    print a

def test_flownet_single(_id=999):
    modelSavePath = os.path.join(base_path, 'trained_models', 'video', 'model_flow.h5')
    return infer_flownet(_id, modelSavePath)

def test_flownet_set(ids, labels, limit=None):
    ## this goes here
    modelSavePath = os.path.join(base_path, 'trained_models', 'video', 'model_flow.h5')
    s = time()
    preds = infer_flownet(ids[:limit], model=modelSavePath)
    print time()-s,'seconds'
    p, r, a = find_precision_recall(labels[:limit], preds)
    print a

