## Can save all the output of trained embedding
## and implement like fastVideo

import __init__
import numpy as np
import os, sys

from utils import get_ids, load_pkl, dump_pkl

base_path = __init__.base_path

testPlots = load_pkl(os.path.join(base_path, "data", "plotFeatures_with_reverse_test.p"))
trainPlots = load_pkl(os.path.join(base_path, "data", "plotFeatures_with_reverse_train.p"))
valPlots = load_pkl(os.path.join(base_path, "data", "plotFeatures_with_reverse_val.p"))

trainIds, valIds, testIds = get_ids()

def transform(plot):
    new_array = [_p[0][::-1][:np.count_nonzero(_p[0])] for _p in plot]
    return new_array

def store_data(ids, plots):
    store_path = os.path.join(base_path, 'data', 'text_sequences')
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    for _id, plot in zip(ids, plots):
        dump_pkl(plot, os.path.join(store_path,'%d.p'%_id))

store_data(trainIds, transform(trainPlots))
store_data(testIds, transform(testPlots))
