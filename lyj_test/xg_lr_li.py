import xgboost as xgb
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import plot_tree
import matplotlib.pyplot as plt
import json
from scipy.sparse import hstack





def get_cun_count(X_train_leaves):
    (rows, cols) = X_train_leaves.shape
    cum_count = np.zeros((1, cols), dtype=np.int32)
    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_train_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_train_leaves[:, j])) + cum_count[0][j - 1]
    return cum_count


def write_leaves(leavesFile,X_train_leaves):
    (rows,cols)=X_train_leaves.shape
    cum_count = get_cun_count(X_train_leaves)
    outfile = open(leavesFile, 'w')
    for j in range(cols):
        if j==0:
            #dat=dict(zip(np.unique(X_train_leaves[:,j]),map(lambda x: 'leaf:'+str(x), range(1,cum_count[0][0]+1))))
            dat=dict(zip(np.unique(X_train_leaves[:,j]),range(10000+1,10000+cum_count[0][0]+1)))
        else :
            dat=dict(zip(np.unique(X_train_leaves[:,j]),range(10000+cum_count[0][j-1]+1,10000+cum_count[0][j]+1)))
        #json.dump(dat,outfile,ensure_ascii=False)
        #outfile.write('\n')
        outfile.write('{0}\t{1}\t\n'.format(j, dat))
        #data=str(dat)
        #outfile.write('{0}\t{1}\t\n'.format(j, data[1:len(data)-1]))
    outfile.close()





def write_leaves_json(leavesFile,X_train_leaves):
    cum_count=get_cun_count(X_train_leaves)
    (rows,cols)=X_train_leaves.shape
    outfile = open(leavesFile, 'w')
    for j in range(cols):
        if j==0:
            dat=dict(zip(map(str,np.unique(X_train_leaves[:,j])),range(1,cum_count[0][0]+1)))
        else :
            dat=dict(zip(map(str,np.unique(X_train_leaves[:,j])),range(cum_count[0][j-1]+1,cum_count[0][j]+1)))
    #json.dump(dat,outfile,ensure_ascii=False)
    #outfile.write('\n')
        json.dump(dat,outfile)
        #outfile.write(a)
        outfile.write('\n')
    outfile.close()



def get_transform_leaves(leavesFile,X_test_leaves,y_test,transformFile):
    # 如果leaves是json文件的话'
    dict=[]
    outfile = open(transformFile, 'w')
    cum_count=get_cun_count(X_test_leaves)

    with open(leavesFile,'r') as inputfile:
        lines=inputfile.readlines()
        for line in lines:
            sA = line.split('\t')
            d = sA[1]
            d1 = eval(d)
            dict.append(d1)
    (rows,cols)=X_test_leaves.shape
    for i in range(rows):
        outfile.write(str(y_test[i]) + '\t')
        for j in range(cols):
            tmp=X_test_leaves[i][j]
            a=dict[j][tmp]
        #outfile.write('{0}:1\t'.format(X_test_leaves[i,:]))
            outfile.write('{0}:1\t'.format(a))
        outfile.write('\n')
    outfile.close()


def get_transform_leaves_json(leavesFile,X_test_leaves,y_test,transformFile):
    dict=[]
    outputfile = open(transformFile, 'w')
    cum_count = get_cun_count(X_test_leaves)
    with open(leavesFile, 'r') as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            lf=json.loads(line)
            dict.append(lf)
    (rows, cols) = X_test_leaves.shape
    for i in range(rows):
        outputfile.write(str(y_test[i]) + '\t')
        for j in range(cols):
            tmp=X_test_leaves[i][j]
            a=dict[j][str(tmp)]
            outputfile.write('{0}:1\t'.format(a))
        outputfile.write('\n')
    outputfile.close()



def xgboost_lr_train(X_train,X_test,y_train,y_test,X_train_leaves,X_test_leaves):
    # lr对原始特征样本模型训练
    lr_ori=LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr_ori.fit(X_train,y_train)
    y_pred_test_ori=lr_ori.predict_proba(X_test)[:,1]
    lr_ori_test_auc=roc_auc_score(y_test,y_pred_test_ori)
    print("original features auc:%.4f"% lr_ori_test_auc)


    # lr对xgboost叶子结点的特征
    get_transform_leaves('../data/leaves_v2.txt', X_train_leaves, y_train, '../data/transform_X_train_leaves_v2')
    get_transform_leaves('../data/leaves_v2.txt', X_test_leaves, y_test, '../data/transform_X_test_leaves_v2')
    X_train_trans,y_train_trans=load_svmlight_file('../data/transform_X_train_leaves_v2')
    X_test_trans,y_test_trans=load_svmlight_file('../data/transform_X_test_leaves_v2',n_features=X_train_trans.shape[1])

    lr_leaves=LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr_leaves.fit(X_train_trans,y_train_trans)
    y_pred_test_leaves=lr_leaves.predict_proba(X_test_trans)[:,1]
    lr_leaves_test_auc=roc_auc_score(y_test_trans,y_pred_test_leaves)
    print("xgboost leaves features auc:%.4f"%lr_leaves_test_auc)

    # origin 特征与叶子结点特征结合在一起
    X_all_train=hstack((X_train,X_train_trans))
    X_all_test=hstack((X_test,X_test_trans))
    lr_mixed=LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr_mixed.fit(X_all_train,y_train)
    y_pred_test_mixed=lr_mixed.predict_proba(X_all_test)[:,1]
    lr_mixed_test_auc=roc_auc_score(y_test,y_pred_test_mixed)
    print("mixed features auc:%.4f"% lr_mixed_test_auc)






def main(X_leaves,y_leaves,X_train,X_test,y_train,y_test,X_train_leaves,X_test_leaves):
    write_leaves('../data/leaves_v2.txt', X_leaves)
    write_leaves_json('../data/leaves_json_v2.txt', X_leaves)
    get_transform_leaves('../data/leaves_v2.txt', X_leaves, y_leaves, '../data/transform_leaves_v2')
    get_transform_leaves_json('../data/leaves_json_v2.txt', X_leaves, y_leaves, '../data/transform_leaves_from_json_v2')
    xgboost_lr_train(X_train, X_test, y_train, y_test, X_train_leaves, X_test_leaves)


if __name__=='__main__':
    train_path = "../data/agaricus.txt.train"
    test_path = "../data/agaricus.txt.test"

    xgtrain = xgb.DMatrix(train_path)
    xgtest = xgb.DMatrix(test_path)

    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    watchlist = [(xgtrain, 'train'), (xgtest, 'test')]

    model_sklearn = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
                                      n_estimators=10, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
    model_sklearn.fit(X_train,y_train)
    params = model_sklearn.get_xgb_params()

    model_xgboost = xgb.train(params, xgtrain, num_boost_round=10, evals=watchlist)

    y_pred_train_sklearn = model_sklearn.predict_proba(X_train)[:, 1]
    y_pred_test_sklearn = model_sklearn.predict_proba(X_test)[:, 1]
    sklearn_train_auc = roc_auc_score(y_train, y_pred_train_sklearn)
    sklearn_test_auc = roc_auc_score(y_test, y_pred_test_sklearn)

    print('sklearn_train_auc:%.4f' % sklearn_train_auc)
    print('sklearn_test_auc:%.4f' % sklearn_test_auc)

    y_pred_train_xgboost = model_xgboost.predict(xgtrain)
    y_pred_test_xgboost = model_xgboost.predict(xgtest)
    # xgboost_train_auc=roc_auc_score(y_train,y_pred_train_xgboost)
    # xgboost_test_auc=roc_auc_score(y_test,y_pred_test_xgboost)

    # print ("xgboost_train_auc:%.4f"% xgboost_train_auc)
    # print ("xgboost_test_auc:%.4f"% xgboost_test_auc)

    xgtrain_leaves = model_xgboost.predict(xgtrain, pred_leaf=True)
    xgtest_leaves = model_xgboost.predict(xgtest, pred_leaf=True)

    X_train_leaves = model_sklearn.apply(X_train)
    X_test_leaves = model_sklearn.apply(X_test)

    #合并之后的全部样本和对应的label
    X_leaves=np.vstack((X_train_leaves,X_test_leaves))
    y_leaves=np.hstack((y_train,y_test))

    # plot_tree(model_sklearn,num_trees=0)
    # plt.show()
    # plot_tree(model_xgboost,num_trees=0)
    # plt.show()

    main(X_leaves,y_leaves,X_train,X_test,y_train,y_test,X_train_leaves,X_test_leaves)




from sklearn.model_selection import GridSearchCV

x=GridSearchCV()