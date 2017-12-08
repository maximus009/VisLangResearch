### Pickles features of individual frames for all videos
## Run python precompute_features.py <model_name>


import os, sys
from pickle import load, dump

from models.visual import remove_last_layer, get_vgg, get_resnet
from video import get_frames

num_samples = 5043

# model_name can be 'vgg' or res'
model_name = "vgg" if len(sys.argv) !=2 else sys.argv[1]

if model_name == "vgg":
    model, preprocess = get_vgg()
if model_name == "res":
    model, preprocess = get_resnet()

model = remove_last_layer(model, 1)

outpath = os.path.join("data", "frameFeatures", "{}Features".format(model_name))

if not os.path.exists(outpath):
    os.makedirs(outpath)

def gather_features():
    for ID in range(num_samples):
        frames = get_frames(ID)
        features = model.predict(preprocess(frames), verbose=1)
        picklePath = os.path.join(outpath,str(ID)+'.p')
        dump(features, open(picklePath, 'wb'))
        print "dumped ",picklePath

gather_features()
