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
import keras.backend as kb
from keras.losses import mean_squared_error

tf_families = config['tf_names'].keys()

df = pd.concat([pd.read_excel(config['publication_data'][tf_family]) for tf_family in tf_families])
tf_names = [config['tf_names'][tf_family] for tf_family in tf_families]
tf_names = [item for sublist in tf_names for item in sublist]

print('TF names: %s' % tf_names)

# missing value represented by -1, its gradient will be masked at training time
df = df.fillna(-1)

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


# custom loss to mask missing value in output


def custom_loss(y_true, y_pred, mask_val=-1):
    # both are 2D array
    # num_examples x num_output
    # find which values in yTrue (target) are the mask value
    is_mask = kb.equal(y_true, mask_val)  # true for all mask values
    is_mask = kb.cast(is_mask, dtype=kb.floatx())
    is_mask = 1 - is_mask  # now mask values are zero, and others are 1
    y_true = y_true * is_mask
    y_pred = y_pred * is_mask
    # TODO reweight to account for proportion of missing value
    # for now we know each example has same number of missing output, so this can be omitted
    return mean_squared_error(y_true, y_pred)


# baseline: fully connected net


def baseline_model(n_in=X.shape[1]*X.shape[2], n_out=Y.shape[1]):
    model = Sequential()
    model.add(Dense(config['training_one_model']['fully_connected']['n_hid'], input_shape=(n_in,),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_out, kernel_initializer='normal'))
    model.compile(loss=custom_loss, optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=baseline_model, epochs=config['training_one_model']['fully_connected']['epochs'],
                           batch_size=config['training_one_model']['fully_connected']['batch_size'], verbose=0)
kfold = KFold(n_splits=config['training_one_model']['fully_connected']['n_folds'], random_state=1234, shuffle=True)
result = cross_validate(estimator, X.reshape([X.shape[0], -1]), Y, cv=kfold,
                        return_estimator=True, return_train_score=True,
                        scoring=('r2', 'neg_mean_squared_error'))
print('Fully connected')
#print('Training r2: ', result['train_r2'])
#print('Validation r2: ', result['test_r2'])
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
    df_plot_2 = df_plot[df_plot[name] != -1]
    corr, pval = pearsonr(df_plot_2[name], df_plot_2['%s_pred' % name])
    fig = df_plot_2.iplot(kind='scatter', x='%s_pred' % name, y=name,
                        xTitle='%s_pred' % name, yTitle=name, title='%s held-out %.4f (%.4e)' % (name, corr, pval),
                        mode='markers', size=1, asFigure=True)
    plotly.offline.plot(fig, filename="report/one_model/%s_fc.html" % name)
    print(corr, pval)
    print(spearmanr(df_plot_2[name], df_plot_2['%s_pred' % name]))


# conv net


def conv_model(n_out=Y.shape[1]):
    model = Sequential()
    for n_filter, filter_width, dilation_rate in config['training_one_model']['conv']['filters']:
        model.add(Conv1D(filters=n_filter, kernel_size=filter_width, strides=1, padding='valid',
                         dilation_rate=dilation_rate, activation='relu', use_bias=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_out, kernel_initializer='normal'))
    model.compile(loss=custom_loss, optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=conv_model, epochs=config['training_one_model']['conv']['epochs'],
                           batch_size=config['training_one_model']['conv']['batch_size'], verbose=0)
kfold = KFold(n_splits=config['training_one_model']['conv']['n_folds'], random_state=1234, shuffle=True)
result = cross_validate(estimator, X, Y, cv=kfold,
                        return_estimator=True, return_train_score=True,
                        scoring=('r2', 'neg_mean_squared_error'))
print('Conv')
# TODO remove these loss
#print('Training r2: ', result['train_r2'])
#print('Validation r2: ', result['test_r2'])
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
    df_plot_2 = df_plot[df_plot[name] != -1]
    corr, pval = pearsonr(df_plot_2[name], df_plot_2['%s_pred' % name])
    fig = df_plot_2.iplot(kind='scatter', x='%s_pred' % name, y=name,
                        xTitle='%s_pred' % name, yTitle=name, title='%s held-out %.4f (%.4e)' % (name, corr, pval),
                        mode='markers', size=1, asFigure=True)
    plotly.offline.plot(fig, filename="report/one_model/%s_conv.html" % name)
    print(corr, pval)
    print(spearmanr(df_plot_2[name], df_plot_2['%s_pred' % name]))









