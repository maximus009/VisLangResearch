import __init__

from models.visual import remove_last_layer, get_vgg, get_resnet
from video import get_frames, preprocess_video
from keras.applications.imagenet_utils import decode_predictions

num_samples = 5043

model, preprocess = get_vgg()
model = remove_last_layer(model, 0)

def gather_features():
    for i in range(num_samples):
        frames = preprocess(get_frames(i))
        features = model.predict(frames, verbose=1)
        frameObjects = decode_predictions(features, top=3)
        for frame in frameObjects[:50]:
            print frame

gather_features()
