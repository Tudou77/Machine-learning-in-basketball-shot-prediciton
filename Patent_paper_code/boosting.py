import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error,precision_score
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
import math


dataset =  pd.read_csv('../shot_logs.csv', header=0)

#target dataset
datasettarget = dataset['FGM']

datasetwithouttarget = dataset[['SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST']]


model = XGBClassifier()
model.fit(datasetwithouttarget,datasettarget)
# plot feature importance
plot_importance(model, importance_type ='weight')
pyplot.show()




# Boosting without parameter tuning
X_train, X_test, y_train, y_test = train_test_split( datasetwithouttarget[['SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST']], datasettarget, test_size=0.05, random_state=42)

xgb_model = XGBClassifier().fit(X_train,y_train)

predictions = xgb_model.predict(X_test)

actuals = y_test

print(confusion_matrix(actuals, predictions))
print(precision_score(actuals, predictions) )




'''
parameters_for_testing = {
    'min_child_weight':[0.0001,0.001,0.01,0.1],
    'learning_rate':[0.00001,0.0001,0.001],
    'n_estimators':[1,2,3,5,10],
    'max_depth':[2,3,4,5]
}
'''
parameters_for_testing = {
    'min_child_weight':[0.0001,0.001,0.01,0.1],
    'learning_rate':[0.0001,0.001],
    'n_estimators':[1,3,5,10],
    'max_depth':[3,4]
}
# Parameter tuning using 7 features
xgb_model = XGBClassifier()

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, scoring='precision')
gsearch1.fit(X_train,y_train)

print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)
