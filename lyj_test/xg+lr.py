import xgboost as xgb
from sklearn.metrics import roc_auc_score
import numpy as np
import datetime

def train_xg(train_path="../data/agaricus.txt.train", test_path="../data/agaricus.txt.test"):

    xgtrain=xgb.DMatrix(train_path)
    xgtest=xgb.DMatrix(test_path)
    y_train=xgtrain.get_label()
    y_test=xgtest.get_label()

    params={
        'max_depth': 5,
        'eta': 0.08,
        'silent': 1,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'subsample': 0.9,
        'colsample_bytree': 0.5,
        'gamma': 0
    }
    watchlist=[(xgtest,'test'),(xgtrain,'train')]
    xgmodel=xgb.train(params, xgtrain, num_boost_round=10, evals=watchlist)
    xgmodel.dump_model('dump.raw.txt')
    xgmodel.save_model('xg.model.' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    y_pred_xgtrain=xgmodel.predict(xgtrain)
    y_pred_xgtest=xgmodel.predict(xgtest)
    xgtrain_auc=roc_auc_score(y_train, y_pred_xgtrain)
    xgtest_auc=roc_auc_score(y_test, y_pred_xgtest)

    print ("xgboost_train_auc:%.4f"% xgtrain_auc)
    print ("xgboost_test_auc:%.4f"% xgtest_auc)

    xgtrain_leaves=xgmodel.predict(xgtrain, pred_leaf=True)
    xgtest_leaves=xgmodel.predict(xgtest, pred_leaf=True)
    print("xgtrain_leaves",xgtrain_leaves, "xgtest_leaves",xgtest_leaves)

    return xgtrain_leaves, xgtest_leaves, y_test


def cum_count_unique(xgtest_leaves):
    (rows, cols)=xgtest_leaves.shape
    cum_count=np.zeros((1,cols),dtype=np.int32)
    for j in range(cols):
        if j==0:
            cum_count[0][j]=len(np.unique(xgtest_leaves[:,0]))
        else:
            cum_count[0][j]=len(np.unique(xgtest_leaves[:,j]))+cum_count[0][j-1]
    print('cum_count',cum_count)
    return cum_count

def save_leaves_map(xgtest_leaves, cum_count, map_path):
    (rows,cols)=xgtest_leaves.shape
    outfile=open(map_path,'w')
    for j in range(cols):
        if j==0:
            dat=dict(zip(np.unique(xgtest_leaves[:,0]), range(1, cum_count[0][0]+1)))
        else:
            dat=dict(zip(np.unique(xgtest_leaves[:,j]), range(cum_count[0][j-1]+1, cum_count[0][j]+1)))
        outfile.write('{0}\t{1}\n'.format(j,dat))
    outfile.close()

def get_transform_leaves(xgtest_leaves, leaves_map, transformFile, y_test):
    dict = []
    outfile = open(transformFile, 'w')
    cum_count = cum_count_unique(xgtest_leaves)

    with open(leaves_map, 'r') as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            sA = line.split('\t')
            d = sA[1]
            d1 = eval(d)
            dict.append(d1)
    (rows, cols) = xgtest_leaves.shape
    for i in range(rows):
        outfile.write(str(y_test[i]) + '\t')
        for j in range(cols):
            tmp = xgtest_leaves[i][j]
            xgtest_leaves[i][j] = dict[j][tmp]
            outfile.write('{0}:1\t'.format(xgtest_leaves[i, j]))
        outfile.write('\n')
    outfile.close()


def main():
    xgtrain_leaves, xgtest_leaves, y_test=train_xg("../data/agaricus.txt.train", "../data/agaricus.txt.test")
    cum_count=cum_count_unique(xgtest_leaves)
    save_leaves_map(xgtest_leaves, cum_count, "../data/leaves_map.txt")
    get_transform_leaves(xgtest_leaves, "../data/leaves_map.txt", "../data/transform_leaves", y_test)

if __name__=="__main__":
    main()