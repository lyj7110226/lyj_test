#!/usr/bin/env python
# xgboost_param_tune
import time
import argparse
import codecs
import xgboost as xgb
import sys
import numpy as np
import datetime


def train_model_for(train_file, test_file):
    dtrain = xgb.DMatrix(train_file)
    dtest = xgb.DMatrix(test_file)
    max_depth_array = [3, 4, 5, 6, 7, 8, 9, 10]
    eta_array = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.20, 0.21, 0.23, 0.25]
    min_child_weight_array = [2, 4, 6, 8, 10, 12]
    subsample_array = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_array = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    lambda_array = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    gamma_array = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    for a in range(len(max_depth_array)):
        for b in range(len(eta_array)):
            for c in range(len(min_child_weight_array)):
                for d in range(len(subsample_array)):
                    for e in range(len(colsample_bytree)):
                        for f in range(len(alpha_array)):
                            for g in range(len(lambda_array)):
                                for h in range(len(gamma_array)):
                                    param = {
                                        'max_depth': max_depth_array[a],
                                        'eta': eta_array[b],
                                        'min_child_weight': min_child_weight_array[c],
                                        'silent': 1,
                                        'objective': 'binary:logistic',
                                        'booster': 'gbtree',
                                        'subsample': subsample_array[d],
                                        'colsample_bytree': colsample_bytree[e],
                                        'alpha': alpha_array[f],
                                        'lambda': lambda_array[g],
                                        'gamma': gamma_array[h],
                                    }
                                    param['eval_metric'] = 'auc'
                                    evallist = [(dtest, 'eval'), (dtrain, 'train')]
                                    label = dtrain.get_label()
                                    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
                                    param['scale_pos_weight'] = ratio
                                    num_round = 200

                                    model_name = 'max_depth:' + str(a) + ' eta:' + str(b) + ' min_child_weight:' + str(
                                        c) + 'subsample: ' + str(d) + 'colsample_bytree: ' + str(e) + ' alpha: ' + str(
                                        f) + 'lambda: ' + str(g) + 'gamma: ' + str(h)
                                    print(model_name)
                                    bst = xgb.train(param, dtrain, num_round, evallist)

                                    #                    bst.dump_model('dump.raw.txt')
                                    #                    bst.save_model('xg.model.' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
                                    print(sorted(bst.get_score(importance_type="gain").items(), key=lambda e: e[1],
                                                 reverse=True))


def predict(model_file):
    xgb.load_model(model_file)


def main(train_file, test_file):
    train_model_for(train_file, test_file)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])