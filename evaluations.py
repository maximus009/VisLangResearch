## write the evaluation functions


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

num_classes = 13
def find_precision_recall(yTrue, yPreds):
    precision = {}
    recall = {}
    avg_pr = {}
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(yTrue[:,i], yPreds[:,i])
        avg_pr[i] = average_precision_score(yTrue[:,i], yPreds[:,i])

    precision['micro'] = precision_recall_curve(yTrue.ravel(), yPreds.ravel())
    avg_pr['micro'] = average_precision_score(yTrue, yPreds, average='micro')
    return precision, recall, avg_pr


