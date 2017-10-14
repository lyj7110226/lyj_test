import xgboost as xgb
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from xgboost import plot_tree
import matplotlib.pyplot as plt
import json

train_path="../data/agaricus.txt.train"
test_path="../data/agaricus.txt.test"

xgtrain=xgb.DMatrix(train_path)
xgtest=xgb.DMatrix(test_path)
y_train=xgtrain.get_label()
y_test=xgtest.get_label()

watchlist=[(xgtrain,'train'),(xgtest,'test')]

params={'eta':0.08, 'max_depth':5, 'gamma':0, 'subsample':0.9, 'colsample_bytree':0.5, 'eval_metric':'auc','silent':1}
model_xgboost=xgb.train(params,xgtrain,num_boost_round=10,evals=watchlist)


y_pred_train_xgboost=model_xgboost.predict(xgtrain)
y_pred_test_xgboost=model_xgboost.predict(xgtest)
xgboost_train_auc=roc_auc_score(y_train,y_pred_train_xgboost)
xgboost_test_auc=roc_auc_score(y_test,y_pred_test_xgboost)
print ("xgboost_train_auc:%.4f"% xgboost_train_auc)
print ("xgboost_test_auc:%.4f"% xgboost_test_auc)


xgtrain_leaves=model_xgboost.predict(xgtrain,pred_leaf=True)
xgtest_leaves=model_xgboost.predict(xgtest,pred_leaf=True)


#plot_tree(model_sklearn,num_trees=0)
##plt.show()
#np.unique(X_train_leaves[:,0])
#plot_tree(model_xgboost,num_trees=0)
#np.unique(xgtrain_leaves[:,0])
##plt.show()

def write_leaves(leavesFile, xgtrain_leaves):
    (rows,cols)=xgtrain_leaves.shape
    cum_count = np.zeros((1, cols), dtype=np.int32)
    outfile = open(leavesFile, 'w')
    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(xgtrain_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(xgtrain_leaves[:, j])) + cum_count[0][j - 1]
    for j in range(cols):
        if j==0:
            dat=dict(zip(np.unique(xgtrain_leaves[:,j]),range(1,cum_count[0][0]+1)))
        else :
            dat=dict(zip(np.unique(xgtrain_leaves[:,j]),range(cum_count[0][j-1]+1,cum_count[0][j]+1)))
        #json.dump(dat,outfile,ensure_ascii=False)
        #outfile.write('\n')
        outfile.write('{0}\t{1}\t\n'.format(j, dat))
    outfile.close()
write_leaves('../data/leaves.txt',xgtrain_leaves)


def write_leaves_svm(leavesFile,X_train_leaves):
    (rows,cols)=X_train_leaves.shape
    cum_count = np.zeros((1, cols), dtype=np.int32)
    outfile = open(leavesFile, 'w')
    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_train_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_train_leaves[:, j])) + cum_count[0][j - 1]
    for j in range(cols):
        if j==0:
            dat=dict(zip(np.unique(X_train_leaves[:,j]),range(1,cum_count[0][0]+1)))
        else :
            dat=dict(zip(np.unique(X_train_leaves[:,j]),range(cum_count[0][j-1]+1,cum_count[0][j]+1)))
        #json.dump(dat,outfile,ensure_ascii=False)
        #outfile.write('\n')
        data=str(dat)
        outfile.write('{0}\t{1}\t\n'.format(j, data[1:len(data)-1]))
    outfile.close()
write_leaves('../data/leaves_svm.txt',xgtrain_leaves)


def write_leaves_json(leavesFile,X_train_leaves):
    (rows,cols)=X_train_leaves.shape
    cum_count = np.zeros((1, cols), dtype=np.int32)
    outfile = open(leavesFile, 'w')
    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_train_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_train_leaves[:, j])) + cum_count[0][j - 1]
    for j in range(cols):
        if j==0:
            dat=dict(zip(np.unique(X_train_leaves[:,j]),range(1,cum_count[0][0]+1)))
        else :
            dat=dict(zip(np.unique(X_train_leaves[:,j]),range(cum_count[0][j-1]+1,cum_count[0][j]+1)))
        #json.dump(dat,outfile,ensure_ascii=False)
        #outfile.write('\n')
        a=json.dumps(dat)
        outfile.write(a)
        outfile.write('\n')
    outfile.close()
#write_leaves('../data/leaves.json',xgtrain_leaves)
write_leaves('../data/leaves', xgtrain_leaves)
