#!/usr/bin/env python
import xgboost as xgb
import sys
import numpy as np
import datetime

def train_model(train_file, test_file, label_score_file):

    dtrain = xgb.DMatrix(train_file)
    dtest = xgb.DMatrix(test_file)

    param = {
         'max_depth': 8,
         'eta': 0.1,
         'silent': 1,
         'objective': 'binary:logistic',
         'booster': 'gbtree',
         'subsample': 0.8,
         'min_child_weight' : 5,
         'colsample_bytree': 0.8,
     'alpha': 5.0,
     'lambda': 5.0,
     'gamma': 0.001
    }
    param['eval_metric'] = 'auc'
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    num_round = 200

    bst = xgb.train( param, dtrain, num_round, evallist )

    bst.dump_model('dump.raw.txt')
    bst.save_model('xg.model.' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    print(sorted(bst.get_score(importance_type="gain").items(), key=lambda e: e[1], reverse=True))

    label2=dtest.get_label()
    ypred_test=bst.predict(dtest)
    print(label2,ypred_test)

    with open(label_score_file, 'w') as f:
        for i in range(len(label2)):
            f.write('{0}\t{1}\n'.format(label2[i],ypred_test[i]))


def predict(model_file):
    xgb.load_model(model_file)

def main(train_file, test_file, label_score_file):
    train_model(train_file, test_file, label_score_file)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
