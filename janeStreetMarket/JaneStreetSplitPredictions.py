from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import kerastuner as kt

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tqdm.auto import tqdm
from tqdm import tqdm
from random import choices

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


# code to feature neutralize

def build_neutralizer(train, features, proportion, return_neut=False):
    """
    Builds neutralzied features, then trains a linear model to predict neutralized features from original
    features and return the coeffs of that model.
    """
    neutralizer = {}
    neutralized_features = np.zeros((train.shape[0], len(features)))
    target = train[['resp', 'bias']].values
    for i, f in enumerate(features):
        # obtain corrected feature
        feature = train[f].values.reshape(-1, 1)
        coeffs = np.linalg.lstsq(target, feature)[0]
        neutralized_features[:, i] = (feature - (proportion * target.dot(coeffs))).squeeze()
        
    # train model to predict corrected features
    neutralizer = np.linalg.lstsq(train[features+['bias']].values, neutralized_features)[0]
    
    if return_neut:
        return neutralized_features, neutralizer
    else:
        return neutralizer

def neutralize_array(array, neutralizer):
    neutralized_array = array.dot(neutralizer)
    return neutralized_array


# def test_neutralization():
#     dummy_train = train.loc[:100000, :]
#     proportion = 1.0
#     neutralized_features, neutralizer = build_neutralizer(dummy_train, features, proportion, True)
#     dummy_neut_train = neutralize_array(dummy_train[features+['bias']].values, neutralizer)
    
# #     assert np.array_equal(neutralized_features, dummy_neut_train)
#     print(neutralized_features[0, :10], dummy_neut_train[0, :10])
    


def neutralize_series(series : pd.Series, by : pd.Series, proportion=1.0):
    """
    neutralize pandas series (originally from the Numerai Tournament)
    """
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)
    exposures = np.hstack((exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

def neutralize(df, target="resp", by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()



# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]



class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1,callbacks=None):
        val_losses = []
        for train_indices, test_indices in splits:
            X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
            y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_test = X_test[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_test = y_test[0]
            
            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train,y_train,
                      validation_data=(X_test,y_test),
                      epochs=epochs,
                        batch_size=batch_size,
                      callbacks=callbacks)
            
            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id, {k:np.mean(val_losses[:,i]) for i,k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


# as indicated by features.csv dataset 
feature_groups_indicators = ["feature_57","feature_58","feature_59","feature_123","feature_3","feature_56","feature_9","feature_129","feature_94","feature_126","feature_75","feature_125","feature_92","feature_55","feature_120","feature_12","feature_21","feature_108","feature_8","feature_27"]
features = ["weight"] + ["feature_" + str(c) for c in range(131)]
f_mean_df1 = np.loadtxt("../input/meansfull/f_mean1")
f_mean_df2 = np.loadtxt("../input/meansfull/f_mean2")
SEED = 42



### Code used for training of the model

# TRAINING = False
# USE_FINETUNE = True     
# FOLDS = 5
# SEED = 42

# train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
# train = train.reset_index(drop = True) 
# train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
# train = train.sort_values(["date"],ascending = False)

# feature_groups_indicators = ["feature_57","feature_58","feature_59","feature_123","feature_3","feature_56","feature_9","feature_129","feature_94","feature_126","feature_75","feature_125","feature_92","feature_55","feature_120","feature_12","feature_21","feature_108","feature_8","feature_27"]
# train['feature_130'] = np.select([train[c].isnull() for c in feature_groups_indicators],list(range(2, len(feature_groups_indicators) + 2)),1)
        
# train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
# train["ts_id"] = train["ts_id"].astype(np.int32)
# train["feature_130"] = train["feature_130"].astype(np.int8)
# train["date"] = train["date"].astype(np.int16)
# train["feature_0"] = train["feature_0"].astype(np.int8)


# train["action"] = ( ( train['resp_1'] + train['resp_2'] + train['resp_3'] + train['resp_4'] + train['resp'] > 0.1  )  ).astype('int')
# train["action"] = train["action"].astype(np.int8)

# features = [c for c in train.columns if 'feature' in c or 'weight' in c]

# df1 = train[train["feature_0"] == 1]
# df2 = train[train["feature_0"] == -1]
    

    
# df1.fillna(df1.mean(),inplace = True)
# df2.fillna(df2.mean(),inplace = True)


# resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

# X1 = df1[features].values
# y1 = np.stack([(df1[c] > 0).astype('int') for c in resp_cols]).T #Multitarget

# X2 = df2[features].values
# y2 = np.stack([(df2[c] > 0).astype('int') for c in resp_cols]).T #Multitarget

# f_mean_df1 = np.mean(df1[features[1:]].values,axis=0)
# f_mean_df2 = np.mean(df2[features[1:]].values,axis=0)




# np.save("f_mean_df_1",f_mean_df1)
# np.save("f_mean_df_2",f_mean_df2)




def create_autoencoder(input_dim,output_dim,noise=0.05):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(64,activation='relu')(encoded)
    decoded = Dropout(0.25)(encoded)
    decoded = Dense(input_dim,name='decoded')(decoded)
    x = Dense(32,activation='relu')(decoded)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(output_dim,activation='sigmoid',name='label_output')(x)
    
    encoder = Model(inputs=i,outputs=encoded)
    autoencoder = Model(inputs=i,outputs=[decoded,x])
    
    autoencoder.compile(optimizer=Adam(0.001),loss={'decoded':'mse','label_output':'binary_crossentropy'})
    return autoencoder, encoder

def create_model(hp,input_dim,output_dim,encoder):
    inputs = Input(input_dim)
    x = inputs
    x = encoder(inputs)
    x = Concatenate()([x,inputs]) #use both raw and encoded features
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('init_dropout',0.0,0.5))(x)
    
    for i in range(hp.Int('num_layers',1,5)):
        x = Dense(hp.Int('num_units_{i}',64,256))(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.swish)(x)
        x = Dropout(hp.Float(f'dropout_{i}',0.0,0.5))(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=x)
    model.compile(optimizer=Adam(hp.Float('lr',0.00001,0.1,default=0.001)),loss=BinaryCrossentropy(label_smoothing=hp.Float('label_smoothing',0.0,0.1)),metrics=[tf.keras.metrics.AUC(name = 'auc')])
    return model




num_of_features = 132
num_of_predictors = 5



models1 = []

# autoencoder1, encoder1 = create_autoencoder(num_of_features,num_of_predictors,noise=0.25)
# encoder1.load_weights('../input/splittest2/encoder_df2.hdf5')
# encoder1.trainable = False
# model_fn1 = lambda hp: create_model(hp,num_of_features,num_of_predictors,encoder1)
# hp = pd.read_pickle(f'../input/splittest1/best_hp_{SEED}_df2.pkl')
# model = model_fn2(hp)
# model.load_weights(f'../input/splittest1/model_{SEED}_df2_finetune.hdf5')
# models2.append(model)

autoencoder1, encoder1 = create_autoencoder(num_of_features,num_of_predictors,noise=0.25)
encoder1.load_weights('../input/splittest2/encoder_df1.hdf5')
encoder1.trainable = False
model_fn1 = lambda hp: create_model(hp,num_of_features,num_of_predictors,encoder1)
hp = pd.read_pickle(f'../input/splittest2/best_hp_{SEED}_df1.pkl')
model = model_fn1(hp)
model.load_weights(f'../input/splittest2/model_{SEED}_df1_finetune.hdf5')
models1.append(model)

autoencoder1, encoder1 = create_autoencoder(num_of_features,num_of_predictors,noise=0.25)
encoder1.load_weights('../input/splittest3/encoder_df1.hdf5')
encoder1.trainable = False
model_fn1 = lambda hp: create_model(hp,num_of_features,num_of_predictors,encoder1)
hp = pd.read_pickle(f'../input/splittest3/best_hp_{SEED}_df1.pkl')
model = model_fn1(hp)
model.load_weights(f'../input/splittest3/model_{SEED}_df1_finetune.hdf5')
models1.append(model)


models2 = []

# autoencoder2, encoder2 = create_autoencoder(num_of_features,num_of_predictors,noise=0.25)
# encoder2.load_weights('../input/splittest2/encoder_df2.hdf5')
# encoder2.trainable = False
# model_fn2 = lambda hp: create_model(hp,num_of_features,num_of_predictors,encoder2)
# hp = pd.read_pickle(f'../input/splittest1/best_hp_{SEED}_df2.pkl')
# model = model_fn2(hp)
# model.load_weights(f'../input/splittest1/model_{SEED}_df2_finetune.hdf5')
# models2.append(model)

autoencoder2, encoder2 = create_autoencoder(num_of_features,num_of_predictors,noise=0.25)
encoder2.load_weights('../input/splittest2/encoder_df2.hdf5')
encoder2.trainable = False
model_fn2 = lambda hp: create_model(hp,num_of_features,num_of_predictors,encoder2)
hp = pd.read_pickle(f'../input/splittest2/best_hp_{SEED}_df2.pkl')
model = model_fn2(hp)
model.load_weights(f'../input/splittest2/model_{SEED}_df2_finetune.hdf5')
models2.append(model)

autoencoder2, encoder2 = create_autoencoder(num_of_features,num_of_predictors,noise=0.25)
encoder2.load_weights('../input/splittest3/encoder_df2.hdf5')
encoder2.trainable = False
model_fn2 = lambda hp: create_model(hp,num_of_features,num_of_predictors,encoder2)
hp = pd.read_pickle(f'../input/splittest3/best_hp_{SEED}_df2.pkl')
model = model_fn2(hp)
model.load_weights(f'../input/splittest3/model_{SEED}_df2_finetune.hdf5')
models2.append(model)





f = np.median
import janestreet
env = janestreet.make_env()
th = 0.5
for (test_df, pred_df) in tqdm(env.iter_test()):
    if test_df['weight'].item() > 0:
        test_df['feature_130'] = np.select([test_df[c].isnull() for c in feature_groups_indicators],list(range(2, len(feature_groups_indicators) + 2)),1)
        x_tt = test_df.loc[:, features].values
        if test_df["feature_0"].item() == 1:
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean_df1
            pred = np.mean([model(x_tt, training = False).numpy() for model in models1],axis=0)
        else:
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean_df2
            pred = np.mean([model(x_tt, training = False).numpy() for model in models2],axis=0)
        pred = f(pred)
        pred_df.action = np.where(pred >= th, 1, 0).astype(int)
    else:
        pred_df.action = 0
    env.predict(pred_df)