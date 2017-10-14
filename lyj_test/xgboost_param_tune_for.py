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
  #  max_depth_array=[5,6,7,8]
  #  eta_array=[0.01,0.1,0.2,0.25]
    min_child_weight_array=[4,6,8]
    subsample_array=[0.5,0.7,0.9]
    colsample_bytree_array=[0.5,0.7,0.9]
    alpha_array=[0.001,0.01,0.1,1,10]
    lambda_array=[0.001,0.01,0.1,1,10]
    gamma_array=[0.001,0.01,0.1,1,10]
    for a in range(len(lambda_array)):
      for b in range(len(alpha_array)):
        for c in range(len(colsample_bytree_array)):
          for d in range(len(subsample_array)):
            for e in range(len(gamma_array)):
              for f in range(len(min_child_weight_array)):
        #        for g in range(len(max_depth_array)):
        #          for h in range(len(eta_array)):
                param = {
                 # 'max_depth': max_depth_array[g],
                 # 'eta': eta_array[h],
                  'eta': 0.25,
                  'max_depth':8,
                  'min_child_weight' : min_child_weight_array[f],
                  'gamma': gamma_array[e],
                  'silent': 1,
                  'objective': 'binary:logistic',
                  'booster': 'gbtree',
                  'subsample': subsample_array[d],
                  'colsample_bytree': colsample_bytree_array[c],
                  'alpha': alpha_array[b],
                  'lambda': lambda_array[a]
                }
                param['eval_metric'] = 'auc'
                evallist  = [(dtest,'eval'), (dtrain,'train')]
                label = dtrain.get_label()
                ratio = float(np.sum(label == 0)) / np.sum(label == 1)
                param['scale_pos_weight'] = ratio
                num_round = 2
                print(param)
                model_name='lambda:'+str(a)+' alpha:'+str(b)+' colsample_bytree:'+str(c)+' subsample:'+str(d)+' gamma:'+str(e)+ ' min_child_weight:'+str(f)
                print(model_name)
                bst = xgb.train( param, dtrain, num_round, evallist )

#                bst.dump_model('dump.raw.txt')
#                bst.save_model('xg.model.' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
                print(sorted(bst.get_score(importance_type="gain").items(), key=lambda e: e[1], reverse=True))

def predict(model_file):
    xgb.load_model(model_file)

def main(train_file, test_file):
    train_model_for(train_file, test_file)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])