## write the evaluation functions
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import os
import numpy as np

from optical.optical_models import spatio_temporal_model
from optical.optical_utils import get_blocks


base_path = os.getcwd()
if not os.path.basename(base_path) == 'moviescope':
    base_path = os.path.join(base_path, '..')

from utils import load_pkl
from keras.models import load_model

## common variables
num_classes = 13


def find_precision_recall(yTrue, yPreds):
    precision = {}
    recall = {}
    avg_pr = {}
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(yTrue[:,i], yPreds[:,i])
        avg_pr[i] = average_precision_score(yTrue[:,i], yPreds[:,i])

    precision['micro'] = precision_recall_curve(yTrue.ravel(), yPreds.ravel())
    avg_pr['micro'] = average_precision_score(yTrue, yPreds, average='micro')
    return precision, recall, avg_pr

def infer_fastVideo(_id, featureName='res', model=None):

    featuresPath = os.path.join(base_path, 'data','frameFeatures','{}Features'.format(featureName))
    if type(_id) == list:
        _multiple = True
    else:
        _multiple = False

    if model is None:
        print "No model passed."
        return
    elif type(model) == str:
        # assuming model path is provided
        model = load_model(model)
    else:
        # reusing preloaded model passed
        pass

    if not _multiple:
        ID = str(_id)
        pklPath = os.path.join(featuresPath, ID+'.p')
        data = load_pkl(pklPath, verbose=True)
        avg = np.mean(data, axis=0)
        input_feature = np.expand_dims(avg, axis=0)
    else:
        input_feature = []
        for ID in _id:
            ID = str(ID)
            pklPath = os.path.join(featuresPath, ID+'.p')
            data = load_pkl(pklPath, verbose=False)
            avg = np.mean(data, axis=0)
            input_feature.append(avg)
        input_feature = np.array(input_feature)

    del featuresPath 

    pred = model.predict(input_feature)
    if _multiple:
        return pred
    else:
        return pred[0]

def infer_flownet(_id, model=None):
    if type(_id) == list:
        _multiple = True
    else:
        _multiple = False

    if model is None:
        print "No model passed."
        return
    elif type(model) == str:
        # always passing a weights file
        model = load_model(model)
    else:
        pass
    xs, xf = [], []
    if not _multiple:
        image, flow, _ = get_blocks(_id)
        xs.append(image)
        xf.append(flow)
    else:
        for block in map(get_blocks, _id):
            image = block[0]
            flow = block[1]
            xs.append(image)
            xf.append(flow)
    xs = np.array(xs)
    xf = np.array(xf)

    input_feature = [xs, xf]
    pred = model.predict(input_feature)
    model.save('model_flow.h5')

    if _multiple:
        return pred
    else:
        return pred[0]


def predict_genres_movie(config):
    ID = config['id']
    type_model = config['type']
    model_name = config['model']

def generate_qualitative_stats(config):
    model_name = config['model_name']


if __name__=="__main__":
    test_fastVideo()
