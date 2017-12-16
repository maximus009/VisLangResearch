import os
from time import time

from evaluations import infer_flownet, infer_fastVideo, find_precision_recall, infer_fastText
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
    return preds, a

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
    return preds, a

def test_fastText_single(_id=999):
    weightsPath = os.path.join(base_path, 'trained_weights', 'text', 'best_weights_glove.h5')
    return infer_fastText(_id, weightsPath)

def test_fastText_set(ids, labels, limit=None):
    weightsPath = os.path.join(base_path, 'trained_weights', 'text', 'best_weights_glove.h5')
    s = time()
    preds = infer_fastText(ids[:limit], weightsPath)
    print time()-s,"seconds"
    p, r, a = find_precision_recall(labels[:limit], preds)
    return preds, a

def late_fusion(load=True):
    from keras.layers import Input, TimeDistributed, Dense, Flatten
    from keras.models import Model, load_model
    from keras.optimizers import SGD
    import numpy as np

    train_limit = 2000
    val_limit = None
    vis_preds_train,_ = test_fastVideo_set(trainIds, trainLabels, limit=train_limit)
    text_preds_train,_ = test_fastText_set(trainIds, trainLabels, limit=train_limit)

    vis_preds_val,_ = test_fastVideo_set(valIds, valLabels, limit=val_limit)
    text_preds_val,_  = test_fastText_set(valIds, valLabels, limit=val_limit)

    train_input = np.dstack((vis_preds_train, text_preds_train))
    val_input = np.dstack((vis_preds_val, text_preds_val))

    #train_input = np.swapaxes(train_input, 1,2)
    #val_input = np.swapaxes(val_input, 1,2)

    if not load:
        input_layer = Input(shape=(13,2))
        out = Dense(13, activation='softmax', use_bias=False)(input_layer)
        out = Flatten()(Dense(1, activation='sigmoid', use_bias=False)(out))
        model = Model(inputs=input_layer, outputs=out)
        model.summary()
    else:
        model = load_model("model_vislang_late_2.h5")

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-3, decay=1e-2, nesterov=True, momentum=0.8))
    model.fit(train_input, trainLabels[:train_limit], validation_data=(val_input, valLabels[:val_limit]), epochs=200)
    model.save('model_vislang_late_2.h5')
    model.save_weights('weights_vislang_late_2.h5')

def test_late_fusion():
    import numpy as np
    from keras.models import load_model

    vis_pred_test, _ = test_fastVideo_set(testIds, testLabels)
    text_pred_test, _ = test_fastText_set(testIds, testLabels)
    test_input = np.dstack((vis_pred_test, text_pred_test))

    model = load_model('model_vislang_late_2.h5')
    preds = model.predict(test_input)

    p, r, a = find_precision_recall(testLabels, preds)
    return a

def modal_attention():
    from keras.models import load_model
    model = load_model('model_vislang_late_2.h5')
    genre_layer = model.layers[1].get_weights()[0].T
    print genre_layer
    #for l in model.layers:
    #    print l.get_weights()

late_fusion()
#print test_late_fusion()
modal_attention()
