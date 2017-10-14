#v1：拿训练集训练xgboost，拿测试集落在的叶子节点当作新的特征，给LR去训练。再拿一份数据集给LR做测试。
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import numpy as np
from xgboost import plot_tree
import datetime
import matplotlib.pyplot as plt

xgtrain=xgb.DMatrix("../data/agaricus.txt.train")
xgtest=xgb.DMatrix("../data/agaricus.txt.test")
y_train=xgtrain.get_label()
y_test=xgtest.get_label()

params = {
    'max_depth': 5,
    'eta': 0.08,
    'silent': 1,
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'subsample': 0.9,
    'colsample_bytree': 0.5,
    'gamma': 0
}
neg_pos=float(np.sum(y_train==0))/np.sum(y_train==1)
params['positive_pos_weight']=neg_pos
watchlist=[(xgtest,'test'),(xgtrain,'train')]
xgmodel=xgb.train(params,xgtrain,num_boost_round=10,evals=watchlist)
#xgmodel.dump_model('dump.raw.txt')
#xgmodel.save_model('xg.model.' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

y_pred_train=xgmodel.predict(xgtrain)
y_pred_test=xgmodel.predict(xgtest)
y_train_auc=roc_auc_score(y_train, y_pred_train)
y_test_auc=roc_auc_score(y_test,y_pred_test)
print("xgtrain_auc: %.4f" %y_train_auc)
print("xgtest_auc: %.4f" %y_test_auc)

train_leaves=xgmodel.predict(xgtrain, pred_leaf=True)
test_leaves=xgmodel.predict(xgtest, pred_leaf=True)

print("train_leaves",train_leaves)
print("test_leaves", test_leaves)

(rows,cols)=test_leaves.shape
cum_count=np.zeros((1,cols),dtype=np.int32)
for j in range(cols):
    if j==0:
        cum_count[0][j]=len(np.unique(test_leaves[:,j]))
    else:
        cum_count[0][j]=len(np.unique(test_leaves[:,j]))+cum_count[0][j-1]

print("cum_count: ",cum_count)

print("第1棵树unique叶子节点",np.unique(train_leaves[:,1]))
print(np.unique(test_leaves[:,1]))

#plot_tree(xgmodel, num_trees=9)
#plt.show()


#对叶子节点进行重新编码，并写入文件

