### store the frame features saved by number of frames per second
## python create_data_batch.py <featureName> <mode>

import __init__

base_path = __init__.base_path

import os, sys
from utils import dump_pkl, load_pkl, get_labels, get_ids


trainIds, valIds, testIds = get_ids()
trainLabels, valLabels, testLabels = get_labels()

# featureName can be 'vgg' or 'res'
featureName = sys.argv[1]
featuresPath = os.path.join(base_path,'data','frameFeatures','{}Features'.format(featureName))

data = {}
for mode in ['train','val','test']:
    ids, labels = {'val':(valIds,valLabels), 'test':(testIds,testLabels), 'train':(trainIds,trainLabels)}[mode]

    for i,ID in enumerate(ids):
        picklePath = os.path.join(featuresPath, str(ID)+'.p')
        frames = load_pkl(picklePath)[::2]
        _l = len(frames)
        l = min([300, _l - _l%25]) # greatest multiple of 25 less than number of frames, or 300, whichever is less

        sample = {'id':ID, 'label':labels[i]}
        if l not in data:
            data[l] = {'train':[], 'val':[], 'test':[]}
            data[l][mode] = [sample]
        else:
            data[l][mode] += [sample]

outPath = os.path.join(base_path,'data','frames_length')
dump_pkl(data,os.path.join(outPath,'{}.p'.format(featureName)))
