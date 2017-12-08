import os
base_path = os.getcwd()
if not os.path.basename(base_path) == 'moviescope':
    base_path = os.path.join(base_path, '..')

frames_resource = os.path.join(base_path, 'frames')
img_width, img_height = 224, 224

import numpy as np
from time import time
from glob import glob
import os

# ID [0:5042]

def get_frames(ID, target_size=(img_width, img_height)):
    from keras.preprocessing.image import load_img, img_to_array

    ID = str(ID) if type(ID) == int else ID
    #assert type(ID) is str, "ID needs to be of type 'str'"
    assert 0 <= int(ID) < 5043, "ID needs to be >=0 and <5043"

    framesPath = os.path.join(frames_resource, ID, '*.png')
    framesList = glob(framesPath)

    print 'Fetching frames from',framesPath
    start = time()
    #frames = np.zeros((len(framesList), img_width, img_height, 3))
    frames = np.array([img_to_array(load_img(frame, target_size=target_size)) for frame in framesList])
    duration = time() - start
    print len(frames), duration,
    speed = len(frames)/duration
    print speed, 'frames per second'
    return frames

def preprocess_video(frames):
    """ zero-centering between -1 and a 1""" 
    """ deprecated. using in-built preprocess """
    frames /= 255.
    frames -= 0.5
    frames *= 2
    return frames

def test_get_frames():

    """ Given a set of IDs, calculate the time taken to retrive the frames """

    indices = map(str, np.random.choice(range(5072), 10))
    print indices
    for i in indices:
        frames = get_frames(i)
