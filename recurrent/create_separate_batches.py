# save batches of data seperately
# note, big_data_{} already has this content, but inefficient when 
# experimenting with differnt batches

import __init__
base_path = __init__.base_path

import os, sys
from utils import load_pkl, dump_pkl
from time import time

featureName = sys.argv[1]
big_data = []

outPath = os.path.join(base_path,'data','big_data_{}.p'.format(featureName))
s = time()
big_data = load_pkl(outPath)
print time()-s,"seconds to load file"

for xt,yt,xv,yv in big_data:
    if xv.tolist() == []:
        continue
    l = xv.shape[1]
    if os.path.exists('batch_data_{}_{}.p'.format(featureName,l)):
        continue
    dump_pkl((xt,yt,xv,yv),'batch_data_{}_{}'.format(featureName,l), verbose=True)
