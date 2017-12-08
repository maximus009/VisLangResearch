from pickle import load, dump

def load_pkl(pklName, verbose=True):

    if not pklName.endswith('.p'):
        pklName += '.p'

    if verbose:
        print "Loading data from", pklName

    return load(open(pklName, 'rb'))

def dump_pkl(data, pklName, verbose=True):

    if not pklName.endswith('.p'):
        pklName += '.p'

    if verbose:
        print "Dumping data to", pklName

    dump(data, open(pklName, 'wb'))
    return

def get_labels():
    import os
    base_path = os.getcwd()
    if not os.path.basename(base_path) == 'moviescope':
        base_path = os.path.join(base_path, '..')

    labelPath = os.path.join(base_path, 'data', 'labels')
    trainLabels = load_pkl(os.path.join(labelPath, 'trainLabels.p'))
    valLabels = load_pkl(os.path.join(labelPath, 'valLabels.p'))
    testLabels = load_pkl(os.path.join(labelPath, 'testLabels.p'))

    return trainLabels, valLabels, testLabels

def get_ids():
    import os
    base_path = os.getcwd()
    if not os.path.basename(base_path) == 'moviescope':
        base_path = os.path.join(base_path, '..')

    indexPath = os.path.join(base_path, 'data', 'index')
    trainIds = load_pkl(os.path.join(indexPath, 'trainIds.p'))
    valIds = load_pkl(os.path.join(indexPath, 'valIds.p'))
    testIds = load_pkl(os.path.join(indexPath, 'testIds.p'))

    return trainIds, valIds, testIds
