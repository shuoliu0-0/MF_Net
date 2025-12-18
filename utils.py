import sys
import torch
import scipy
import numpy
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def metrics_reg(targets,predicts):
    mae = metrics.mean_absolute_error(y_true=targets,y_pred=predicts)
    rmse = metrics.mean_squared_error(y_true=targets,y_pred=predicts,squared=False)
    pr = scipy.stats.mstats.pearsonr(targets, predicts)[0]
    r2 = metrics.r2_score(targets, predicts)

    x = [ [item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x,y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

    return [mae,rmse,pr,sd,r2]