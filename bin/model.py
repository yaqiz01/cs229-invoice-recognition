from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import BernoulliNB 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import f1_score
import argparse
import random
import numpy as np

# rows: actual classes
# cols: predicted classes
#      A  B  C                  
#  A   TP E  E                      
#  B   E  TP E                  
#  C   E  E  TP                 
def confmat(classes, y_pred, y_test):
    cm = {} 
    for clr in classes:
        cm[clr] = {}
        for clc in classes:
            cm[clr][clc] = 0

    for yp, yt in zip(y_pred, y_test):
        cm[yt][yp] += 1 

    cmmat = np.zeros((len(classes), len(classes)), dtype=np.int32)
    for i, clr in enumerate(classes):
        for j, clc in enumerate(classes):
            cmmat[i,j] = cm[clr][clc]

    # print(cmmat)
    return cm

# sum of values in the corresponding row except the TP
def FN(cl, cm):
    fn = 0
    for clc in cm.keys():
        if clc != cl:
            fn += cm[cl][clc]
    return fn

# sum of values in the corresponding column except the TP
def FP(cl, cm):
    fp = 0
    for clr in cm.keys():
        if clr != cl:
            fp += cm[clr][cl]
    return fp

def TN(cl, cm):
    tn = 0
    for clr in cm.keys():
        for clc in cm.keys():
            if clr != cl or clc !=cl:
                tn += cm[clr][clc]
    return tn

def TP(cl, cm):
    return cm[cl][cl]

def errmetric(classes, y_test, y_pred): 
    err = (y_test != y_pred).sum()
    accuracy = err * 1.0 / y_test.size
    cm = confmat(classes, y_pred, y_test)
    precision = {}
    for cl in classes:
        tp = float(TP(cl, cm))
        fp = float(FP(cl, cm))
        fn = float(FN(cl, cm))
        if (tp + fp) == 0:
            if (tp + fn) == 0:
                precision[cl] = 1 
            else:
                precision[cl] = 0 
        else:
            precision[cl] = tp/(tp + fp)
    recall = {}
    for cl in classes:
        tp = float(TP(cl, cm))
        fn = float(FN(cl, cm))
        recall[cl] = tp/(tp + fn)
        # if (tp + fn)==0:
            # if tp==0:
                # recall[cl] = 1 
            # else:
                # recall[cl] = 0 
        # else:
            # recall[cl] = tp/(tp + fn)
    specificity = {}
    for cl in classes:
        tn = float(TN(cl, cm))
        fp = float(FP(cl, cm))
        if (tn + fp)==0:
            specificity[cl] = -1
        else:
            specificity[cl] = tn/(tn + fp)
    metric = {}
    metric['accuracy'] = accuracy 
    metric['recall'] = recall 
    metric['precision'] = precision 
    metric['specificity'] = specificity 

    yts = []
    yps = []
    for yt, yp in zip(y_test,y_pred):
        if yt!=0:
            yts.append(yt)
            yps.append(yp)
    metric['f1-wavg'] = f1_score(yts, yps, average='weighted')  
    return metric 

def printMetric(metric):
    print('accuracy:{0}'.format(metric['accuracy']))
    for cl in metric['recall'].keys():
        print('{0} recall:{1} precision:{2} specificity:{3}'.format(cl, metric['recall'][cl],
            metric['precision'][cl], metric['specificity'][cl]))

def test(X, y, **opts):
    algo = opts['algo']
    C = opts['C']
    k = opts['k']
    penalty = opts['penalty']

    if penalty=='l1':
        dual = False
    elif penalty=='l2':
        dual = True

    if algo == 'nb':
        md = BernoulliNB()
    elif algo == 'svm':
        md = svm.LinearSVC(multi_class='ovr', penalty=penalty, C=C, dual=dual)
        # md = svm.LinearSVC(multi_class='crammer_singer')
    elif algo == 'logreg':
        md = LogisticRegression(multi_class='ovr', penalty=penalty, C=C, dual=dual)

    classes = sorted(list(set(y)))

    if (opts['mode']=='TRAINING_ERR'):
        X_train = X.copy()
        y_train = y.copy()
        X_test = X.copy()
        y_test = y.copy()
        y_pred = md.fit(X_train, y_train).predict(X_test)
        metric = errmetric(classes, y_test, y_pred)
    elif (opts['mode']=='TESTING_ERR'):
        # mask = []
        # for i in xrange(0, y.size):
            # mask.append(random.randint(0, 9) < 8) # 80% train and 20% test
        # mask = np.array(mask)
        # X_train = X[mask]
        # y_train = y[mask]
        # X_test = X[~mask]
        # y_test = y[~mask]
        split_point = int(0.3*y.size)
        X_train = X[:split_point]
        y_train = y[:split_point]
        X_test = X[split_point:]
        y_test = y[split_point:]

        y_pred = md.fit(X_train, y_train).predict(X_test)
        metric = errmetric(classes, y_test, y_pred)
    elif (opts['mode']=='LEAVE-ONE-OUT'):
        ones = [True] * y.size 
        yts = []
        yps = []
        for i in xrange(0, y.size): 
            mask = np.array(ones)
            mask[i] = False
            X_train = X[mask]
            y_train = y[mask]
            X_test = X[~mask]
            y_test = y[~mask]
            y_pred = md.fit(X_train, y_train).predict(X_test)
            yts.append(y_test[0])
            yps.append(y_pred[0])
        metric = errmetric(classes, np.array(yts), np.array(yps))
    elif (opts['mode']=='K-FOLD'):
        ones = [True] * y.size 
        yts = []
        yps = []
        for i in xrange(0, k): 
            mask = np.array(ones)
            step = y.size/k
            sptr = i*step
            if i==k-1:
                eptr = y.size-1
            else:
                eptr = sptr + step 
            mask[sptr:eptr] = False
            X_train = X[mask]
            y_train = y[mask]
            X_test = X[~mask]
            y_test = y[~mask]
            y_pred = md.fit(X_train, y_train).predict(X_test)
            yts += list(y_test)
            yps += list(y_pred)
        metric = errmetric(classes, np.array(yts), np.array(yps))
    return metric

def main():
    usage = "Usage: model [options --algo]"
    parser = argparse.ArgumentParser(description='Train multiclass model')
    parser.add_argument('--mode', dest='mode', action='store', default='TESTING_ERR',
            help='Testing mode')
    parser.add_argument('--algo', dest='algo', action='store', default='naive_bayes',
            help='algorithm for the model')
    parser.add_argument('--k', dest='k', nargs='?', default=3, type=int,
            help='k for K-Fold')

    (opts, args) = parser.parse_known_args()
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    err = model(X, y, **vars(opts))
    
if __name__ == "__main__":
    main()
