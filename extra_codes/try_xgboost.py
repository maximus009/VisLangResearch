import __init__
base_path = __init__.base_path

import numpy as np
from pandas import read_csv 

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix


from utils import load_pkl, get_labels, get_ids

metadata_path = os.path.join(base_path,'data','movie_metadata.csv')
inFile = open(metadata_path)
data = read_csv(inFile)

attributes = [
        'num_critic_for_reviews',
        'duration',
        'actor_1_facebook_likes',
        'actor_2_facebook_likes',
        'actor_3_facebook_likes',
        'num_voted_users',
        'facenumber_in_poster',
        'num_user_for_reviews',
        'imdb_score',
        'title_year',
        'movie_facebook_likes'
        ]

matrix = data.fillna(0).as_matrix(columns=attributes)

trainLabels, valLabels, testLabels = get_labels()
trainIds, valIds, testIds = get_ids()

xTrain = matrix[trainIds]
xVal = matrix[valIds]
xTest = matrix[testIds]

print xTrain.shape, xVal.shape

#model = RF(n_jobs=4)
#model.fit(xTrain, trainLabels)
#valPredictions = model.predict(xVal)
import xgboost

print trainLabels[:,1].shape

losses = []
eval_metrics = ['error', 'logloss']
for c in range(valLabels.shape[1]):

    model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=500)
    eval_set = [(xVal, valLabels[:,c])]
    model.fit(xTrain, trainLabels[:,c], eval_metric=['error','auc','logloss'], eval_set=eval_set, verbose=True)
    res = model.evals_result()['validation_0']
    losses.append(res)

for loss_class in losses:
    print np.mean(loss_class['auc'])
