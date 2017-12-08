import __init__

from evaluations import find_precision_recall
from glob import glob
from time import time
from multiprocessing import Pool
import numpy as np
import os, sys, cv2, gc

from utils import get_labels, get_ids
from models.visual import get_resnet, remove_last_layer, get_temporal

from keras.layers import Input, Average, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam

num_classes = 13

def spatial_model():
    res_model, preprocess = get_resnet(include_top=False)
    for layer in res_model.layers:
        layer.trainable = False
    return res_model

def spatio_temporal_model():
    image_input_layer = Input(shape=(224,224,3))
    flow_input_layer = Input(shape=(224,224,20))

    spatialModel = spatial_model()
    temporalModel = get_temporal(flow_input_layer)

    spatialModel = spatialModel(image_input_layer)
    spatioTemporalModel = Average(name="merge_avg")([spatialModel, temporalModel])


    spatioTemporalModel = Flatten()(spatioTemporalModel)
    dense = Dense(128, activation='relu')(spatioTemporalModel)
    dense = Dropout(0.5)(dense)
    predictions = Dense(num_classes, activation='sigmoid')(dense)

    stModel = Model(inputs=[image_input_layer, flow_input_layer], outputs=predictions)

    return stModel

def get_only_spatial_model():
    input_layer = Input(shape=(224,224,3))
    res_model, preprocess = get_resnet()
    res_model = remove_last_layer(res_model,1)
    dense = Dropout(0.5)(Dense(512, activation='relu')(res_model(input_layer)))
    out = Dense(num_classes,activation='sigmoid')(dense)
    sModel = Model(inputs=input_layer,outputs=out)
    return sModel


trainIds, valIds, testIds = get_ids()
trainLabels, valLabels, testLabels = get_labels()

base_path = __init__.base_path
flow_image_path = os.path.join(base_path,'data','flow')


###
# read a batch of videos
# for each video, randomly sample a frame
# retrieve the corresponding + 10 optical frames = 20 channels
# pass image through spatial net
# pass the optical flow through temporal net
# sigmoid plus backprop
###


def get_blocks(arg):#ID, label):
    """ more efficient implementation """
    ID,label = arg[0],arg[1]
    ## single movie
    ## returns blocks of flow, flow for t-5 to t+5 of mid image at time t, and label
    paths = sorted(glob(os.path.join(flow_image_path,str(ID),'repr_*')))
    data_flow_blocks = []
    flow_stack = []
    start_index = np.random.choice(len(paths)-11)
    for index in range(start_index, start_index+10):
        path = paths[index]
        repr_image = cv2.imread(path)
        _, rgb_image, flow_h, flow_v = np.hsplit(repr_image,4)

        flow_h = np.mean(flow_h, axis=-1)
        flow_v = np.mean(flow_v, axis=-1)


        flow_stack.append(flow_h)
        flow_stack.append(flow_v)

        if index == start_index+5:
            # to select the middle frame in the block
            mid_image = rgb_image

    if np.random.choice([False,True]):
        # 50=50 chance of flipping the image and the corresponding flow block
        mid_image = np.fliplr(mid_image)
        for _flow in flow_stack:
            _flow = np.fliplr(_flow)

    flow = np.zeros((224,224,20))
    for i,f in enumerate(flow_stack):
        flow[:,:,i] = f

    return (mid_image, flow, label)

def test_get_blocks():
    pobj = Pool(8)
    s = time()
    ## multiprocessing implementation to gather frame/flows of batch data
    pobj.map(get_blocks, zip(valIds,valLabels))
    print time()-s,"seconds for val"

    s = time()
    pobj.map(get_blocks, zip(trainIds,trainLabels))
    print time()-s,"seconds for train"

def get_batches(mode='train',batch_size=32):
    p = Pool(8)
    if mode == 'train':
        ids_and_labels = zip(trainIds,trainLabels)
    elif mode == 'val':
        ids_and_labels = zip(valIds,valLabels)
    elif mode == 'test':
        ids_and_labels = zip(testIds,testLabels)

    L = len(ids_and_labels)
    if batch_size in [-1,"all"]:
        batch_size = L

    for i in range(0,L,batch_size):
        blocks = p.map(get_blocks,ids_and_labels[i:i+batch_size])
        yield blocks

    return


def train_spatial(model_name):

    model = get_only_spatial_model()
    model.summary()

    weightsPath = 'weights_'+model_name+'.h5'
    if os.path.exists(weightsPath):
        model.load_weights(weightsPath)

    optim = Adam(lr=0.0005,decay=0.01)
    model.compile(loss="binary_crossentropy",optimizer=optim)
    epochs = 50 

    print 'Gathering Fixed Val set.'
    blocks = list(get_batches(mode='val',batch_size='all'))[0]
    xv = []
    yv = []
    for block in blocks:
        xv.append(block[0])
        yv.append(block[2])
    xv = np.array(xv)
    yv = np.array(yv)
    print 'done.'

    best_ap = 0.0

    with open('experiment_stats/%s' % model_name, 'a') as statFile:
        statFile.write('\nExperiment %s' % model_name)

    for epoch in range(1,epochs+1):
        print 'Starting epoch:',epoch
        start = time()
        pred = None
        y_actual = None
        for batch_num,batch_blocks in enumerate(get_batches(mode='train',batch_size=32)):
            print 'Loading',batch_num,'\r',
            x = []
            y = []
            for block in batch_blocks:
                x.append(block[0])
                y.append(block[2])
            x = np.array(x)
            y = np.array(y)

            model.train_on_batch(x,y)

            if pred is None:
                pred = model.predict(x)
                y_actual = y
            else:
                pred = np.concatenate((pred, model.predict(x)))
                y_actual = np.concatenate((y_actual, y))
        ## evaluation
        precision, recall, meanap = find_precision_recall(y_actual,pred)
        print '\ntrain ap',meanap['micro'],
        train_ap = meanap['micro']

        pred = None
        y_actual = None
        for batch_num,batch_blocks in enumerate(get_batches(mode='val',batch_size=32)):
            print 'Evaluating',batch_num,'\r',
            x = []
            y = []

            for block in batch_blocks:
                x.append(block[0])
                y.append(block[2])
            x = np.array(x)
            y = np.array(y)
            if pred is None:
                pred = model.predict(x)
                y_actual = y
            else:
                pred = np.concatenate((pred, model.predict(x)))
                y_actual = np.concatenate((y_actual, y))
        precision, recall, meanap = find_precision_recall(y_actual,pred)
        print 'val ap',meanap['micro'],
        val_ap = meanap['micro']

        pred = model.predict(xv)
        precision, recall, meanap = find_precision_recall(yv,pred)
        print 'Fixed val ap:',meanap['micro'],
        fixed_ap = meanap['micro']

        if fixed_ap > best_ap:
            model.save_weights('weights_'+model_name+'.h5')
            best_ap = fixed_ap

        epochTime = time()-start
        print epochTime,"seconds."

        with open('experiment_stats/%s' % model_name, 'a') as statFile:
            statFile.write('\nepoch:{},train_ap:{},val_ap:{},fixed_ap:{},time_s:{}\n'.format(epoch,train_ap,val_ap,fixed_ap,epochTime))

        gc.collect()

    statFile.close()

def train_flow(model_name):

    model = spatio_temporal_model()
    model.summary()

    weightsPath = 'weights_'+model_name+'.h5'
    if os.path.exists(weightsPath):
        model.load_weights(weightsPath)

    model.compile(loss="binary_crossentropy",optimizer='adam')
    epochs = 100

    print 'Gathering Fixed Val set.'
    blocks = list(get_batches(mode='val',batch_size='all'))[0]
    xs_v = []
    xf_v = []
    y = []

    for block in blocks:
        xs_v.append(block[0])
        xf_v.append(block[1])
        y.append(block[2])
    xs_v = np.array(xs_v)
    xf_v = np.array(xf_v)
    yv = np.array(y)
    #xs_v = np.array([block[0] for block in blocks])
    #xf_v = np.array([block[1] for block in blocks])
    #yv = np.array([block[2] for block in blocks])
    print 'done.'

    model_name = 'flow'
    best_ap = 0.0

    with open('experiment_stats/%s' % model_name, 'a') as statFile:
        statFile.write('\nExperiment %s' % model_name)

    for epoch in range(1,epochs+1):
        print 'Starting epoch:',epoch
        start = time()
        pred = None
        y_actual = None
        for batch_num,batch_blocks in enumerate(get_batches(mode='train',batch_size=32)):
            print 'Loading',batch_num,'\r',
            #xs = np.array([block[0] for block in batch_blocks])
            #xf = np.array([block[1] for block in batch_blocks])
            #y = np.array([block[2] for block in batch_blocks])
            xs = []
            xf = []
            y = []
            for block in batch_blocks:
                xs.append(block[0])
                xf.append(block[1])
                y.append(block[2])
            xs = np.array(xs)
            xf = np.array(xf)
            y = np.array(y)

            model.train_on_batch([xs,xf],y)

            if pred is None:
                pred = model.predict([xs,xf])
                y_actual = y
            else:
                pred = np.concatenate((pred, model.predict([xs,xf])))
                y_actual = np.concatenate((y_actual, y))
        ## evaluation
        precision, recall, meanap = find_precision_recall(y_actual,pred)
        print '\ntrain ap',meanap['micro'],
        train_ap = meanap['micro']

        pred = None
        y_actual = None
        for batch_num,batch_blocks in enumerate(get_batches(mode='val',batch_size=32)):
            print 'Evaluating',batch_num,'\r',
            #xs = np.array([block[0] for block in batch_blocks])
            #xf = np.array([block[1] for block in batch_blocks])
            #y = np.array([block[2] for block in batch_blocks])
            xs = []
            xf = []
            y = []
            for block in batch_blocks:
                xs.append(block[0])
                xf.append(block[1])
                y.append(block[2])
            xs = np.array(xs)
            xf = np.array(xf)
            y = np.array(y)
            if pred is None:
                pred = model.predict([xs,xf])
                y_actual = y
            else:
                pred = np.concatenate((pred, model.predict([xs,xf])))
                y_actual = np.concatenate((y_actual, y))
        precision, recall, meanap = find_precision_recall(y_actual,pred)
        print 'val ap',meanap['micro'],
        val_ap = meanap['micro']

        pred = model.predict([xs_v,xf_v])
        precision, recall, meanap = find_precision_recall(yv,pred)
        print 'Fixed val ap:',meanap['micro'],
        fixed_ap = meanap['micro']

        if fixed_ap > best_ap:
            model.save_weights('weights_'+model_name+'.h5')
            best_ap = fixed_ap

        epochTime = time()-start
        print epochTime,"seconds."

        with open('experiment_stats/%s' % model_name, 'a') as statFile:
            statFile.write('\nepoch:{},train_ap:{},val_ap:{},fixed_ap:{},time_s:{}\n'.format(epoch,train_ap,val_ap,fixed_ap,epochTime))

        gc.collect()

    statFile.close()


def main():

    model_name = sys.argv[1]
    if model_name not in ['spatial','flow']:
        print 'ModelNameError:',model_name
        return
    if model_name == 'flow':
        train_flow(model_name)
    else:
        train_spatial(model_name)

if __name__=="__main__":
    print 'Training net for',sys.argv[1]
    main()
