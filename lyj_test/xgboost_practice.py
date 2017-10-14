import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import  GridSearchCV
from sklearn.datasets import   load_svmlight_file


def modelfit(alg, dtrain, dtest,xgtrain, cv_folds=5, early_stopping_rounds=50):

    xgb_param = alg.get_xgb_params()
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=1)
    alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[0], dtrain[1], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[0])
    dtrain_predprob = alg.predict_proba(dtrain[0])[:, 1]
    dtest_predictions = alg.predict(dtest[0])
    dtest_predprob = alg.predict_proba(dtest[0])[:, 1]



    train_accuracy = metrics.accuracy_score(dtrain[1], dtrain_predictions)
    test_accuracy=metrics.accuracy_score(dtest[1],dtest_predictions)
    train_auc=metrics.roc_auc_score(dtrain[1], dtrain_predprob)
    test_auc=metrics.roc_auc_score(dtest[1],dtest_predprob)

    # Print model report:
    print ("\nModel Report")
    print ("Accuracy (Train): %.4g" % train_accuracy)
    print ("Accuracy (Test): %.4g" % test_accuracy)
    print ("Auc Score (Train): %f" % train_auc)
    print ("Auc Score (Test): %f" % test_auc)
    print(cvresult)
    print("\n")
    return train_accuracy,test_accuracy,train_auc,test_auc,cvresult.shape[0]

def tune(estimator,param_test,dtrain):

    gsearch1=GridSearchCV(estimator=estimator,param_grid=param_test,scoring='roc_auc',n_jobs=4,iid=False,cv=5)

    x_train=dtrain[0]
    y_train=dtrain[1]
    gsearch1.fit(x_train,y_train)

    print (gsearch1.grid_scores_)
    print (gsearch1.best_params_)
    print (gsearch1.best_score_)
    return gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_
def base(xgb,dtrain,dtest,xgtrain):

    learning_rate= [0.01, 0.1, 0.2, 0.25, 0.3]
    for eta in learning_rate:
        res=modelfit(xgb,dtrain,dtest,xgtrain)
    return res

def main():
    # Load data for XGBClassifier
    dtrain = load_svmlight_file("../data/agaricus.txt.train")
    dtest = load_svmlight_file("../data/agaricus.txt.test")

    # load data for xgboost
    xgtrain = xgb.DMatrix("../data/agaricus.txt.train")
    #xgtest = xgb.DMatrix("../data/agaricus.txt.test")

    xgb0 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=2,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
#   res0=modelfit(xgb0, dtrain, dtest, xgtrain)
#    xgb0.set_params(n_estimators=res0[4])

    res00=base(xgb0,dtrain,dtest,xgtrain)



    param_test1 = {
        'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }
    gridSearch_res1=tune(xgb0,param_test=param_test1,dtrain=dtrain)
    xgb0.set_params(max_depth=gridSearch_res1[1]['max_depth'],min_child_weight=gridSearch_res1[1]['min_child_weight'])
    res1=modelfit(xgb0,dtrain,dtest,xgtrain)
    print(res1)

    param_test2={
        'gamma':[i/10 for i in range(0,5)]
    }
    gridSearch_res2 = tune(xgb0, param_test=param_test2,dtrain=dtrain)
    xgb0.set_params(gamma=gridSearch_res2[1]['gamma'])
    res2 = modelfit(xgb0, dtrain, dtest, xgtrain)

    param_test3 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gridSearch_res3 = tune(xgb0, param_test=param_test3,dtrain=dtrain)
    xgb0.set_params(subsample=gridSearch_res3[1]['subsample'],colsample_bytree=gridSearch_res3[1]['colsample_bytree'])
    res3 = modelfit(xgb0, dtrain, dtest, xgtrain)

    #param_test4 = {
    #    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    #    'reg_lambda': []
    #}
    #gridSearch_res4 = tune(xgb0, param_test=param_test4)
    #xgb0.set_params(reg_alpha=gridSearch_res4[1]['reg_alpha'], reg_lambda=gridSearch_res1[1]['reg_lambda'])
    #res4 = modelfit(xgb0, dtrain, dtest, xgtrain)

if __name__=="__main__":
    main()