import sys
import matplotlib
matplotlib.use('Agg')   # do not remove, this is to turn off X server so plot works on Linux
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(color_codes=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='ggplot')
from scipy.stats import pearsonr, spearmanr
import plotly.offline
from config import config
import os
import sys

print os.environ['CONDA_DEFAULT_ENV']
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['training']['gpu_id'])

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import regularizers

tf_family = sys.argv[1]
assert tf_family in config['tf_names']

df = pd.read_excel(config['publication_data'][tf_family])
tf_names = config['tf_names'][tf_family]

print('TF names: %s' % tf_names)

# make dataset
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
_data_input = []
_data_output = []
for i, row in df.iterrows():
    seq = row['Sequence']
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
    x = np.asarray(map(int, list(seq)))
    x = IN_MAP[x.astype('int8')]
    _data_input.append(x)

    _val = [row[name] for name in tf_names]
    _data_output.append(_val)

X = np.asarray(_data_input)
Y = np.asarray(_data_output)

print('Encoded data: ', X.shape, Y.shape)

# baseline: fully connected net


def baseline_model(n_in=X.shape[1]*X.shape[2], n_out=Y.shape[1]):
    model = Sequential()
    model.add(Dense(config['training']['fully_connected']['n_hid'], input_shape=(n_in,),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_out, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=baseline_model, epochs=config['training']['fully_connected']['epochs'],
                           batch_size=config['training']['fully_connected']['batch_size'], verbose=2)
kfold = KFold(n_splits=config['training']['fully_connected']['n_folds'], random_state=1234, shuffle=True)
result = cross_validate(estimator, X.reshape([X.shape[0], -1]), Y, cv=kfold,
                        return_estimator=True, return_train_score=True,
                        scoring=('r2', 'neg_mean_squared_error'))
print('Fully connected')
print('Training r2: ', result['train_r2'])
print('Validation r2: ', result['test_r2'])
# make prediction
y_pred = np.empty(Y.shape)
for i, (train_index, test_index) in enumerate(kfold.split(X.reshape([X.shape[0], -1]))):
    xt = X.reshape([X.shape[0], -1])[test_index, :]
    y_pred[test_index, :] = result['estimator'][i].predict(xt)
_data = dict()
for i, name in enumerate(tf_names):
    _data[name] = df[name]
    _data['%s_pred' % name] = y_pred[:, i]
df_plot = pd.DataFrame(_data)

# plot, metric
for name in tf_names:
    corr, pval = pearsonr(df_plot[name], df_plot['%s_pred' % name])
    fig = df_plot.iplot(kind='scatter', x='%s_pred' % name, y=name,
                        xTitle='%s_pred' % name, yTitle=name, title='%s held-out %.4f (%.4e)' % (name, corr, pval),
                        mode='markers', size=1, asFigure=True)
    plotly.offline.plot(fig, filename="report/%s_%s_fc.html" % (tf_family, name))
    print(corr, pval)
    print(spearmanr(df_plot[name], df_plot['%s_pred' % name]))


# conv net


def conv_model(n_out=Y.shape[1]):
    model = Sequential()
    for n_filter, filter_width, dilation_rate in config['training']['conv']['filters']:
        model.add(Conv1D(filters=n_filter, kernel_size=filter_width, strides=1, padding='valid',
                         dilation_rate=dilation_rate, activation='relu', use_bias=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_out, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=conv_model, epochs=config['training']['conv']['epochs'],
                           batch_size=config['training']['conv']['batch_size'], verbose=2)
kfold = KFold(n_splits=config['training']['conv']['n_folds'], random_state=1234, shuffle=True)
result = cross_validate(estimator, X, Y, cv=kfold,
                        return_estimator=True, return_train_score=True,
                        scoring=('r2', 'neg_mean_squared_error'))
print('Conv')
print('Training r2: ', result['train_r2'])
print('Validation r2: ', result['test_r2'])
# make prediction
y_pred = np.empty(Y.shape)
for i, (train_index, test_index) in enumerate(kfold.split(X)):
    xt = X[test_index, :, :]
    y_pred[test_index, :] = result['estimator'][i].predict(xt)
_data = dict()
for i, name in enumerate(tf_names):
    _data[name] = df[name]
    _data['%s_pred' % name] = y_pred[:, i]
df_plot = pd.DataFrame(_data)

# plot, metric
for name in tf_names:
    corr, pval = pearsonr(df_plot[name], df_plot['%s_pred' % name])
    fig = df_plot.iplot(kind='scatter', x='%s_pred' % name, y=name,
                        xTitle='%s_pred' % name, yTitle=name, title='%s held-out %.4f (%.4e)' % (name, corr, pval),
                        mode='markers', size=1, asFigure=True)
    plotly.offline.plot(fig, filename="report/%s_%s_conv.html" % (tf_family, name))
    print(corr, pval)
    print(spearmanr(df_plot[name], df_plot['%s_pred' % name]))









