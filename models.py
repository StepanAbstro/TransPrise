"""
TransPrise model for learning and prediction
"""

import keras.layers as l
from keras import backend
from keras.engine.topology import Container
from keras.models import Sequential


def classification_model(data_shape):
    """
    
    :param data_shape: input data shape (length and layers)
    :return: models for classification
    """

    class_mod = Sequential()
    class_mod.add(l.BatchNormalization(axis=1, input_shape=data_shape))
    class_mod.add(l.Conv1D(16, 64, padding='same', activation='elu'))
    class_mod.add(l.BatchNormalization(axis=1))
    class_mod.add(l.Conv1D(32, 16, padding='same', activation='elu'))
    class_mod.add(l.BatchNormalization(axis=1))
    class_mod.add(l.AveragePooling1D(32))
    class_mod.add(l.Dropout(0.5))
    class_mod.add(l.Flatten())
    class_mod.add(l.Dense(128, activation='elu'))
    class_mod.add(l.Dense(128, activation='elu'))
    class_mod.add(l.BatchNormalization(axis=1))
    class_mod.add(l.Dropout(0.5))
    class_mod.add(l.Dense(1, activation='sigmoid'))

    class_mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return class_mod


def regression_model(data_shape):
    """
    
    :param data_shape: input data shape (length and layers)
    :return: models for regression
    """
    print(data_shape)
    regr_mod = Sequential()
    regr_mod.add(l.BatchNormalization(axis=1, input_shape=data_shape))
    regr_mod.add(l.Conv1D(16, 32, padding='same', activation='elu'))
    regr_mod.add(l.BatchNormalization(axis=1))
    regr_mod.add(l.Conv1D(32, 32, padding='same', activation='elu'))
    regr_mod.add(l.BatchNormalization(axis=1))
    regr_mod.add(l.AveragePooling1D(32))
    regr_mod.add(l.Dropout(0.5))
    regr_mod.add(l.Flatten())
    regr_mod.add(l.Dense(128, activation='elu'))
    regr_mod.add(l.Dense(128, activation='elu'))
    regr_mod.add(l.BatchNormalization(axis=1))
    regr_mod.add(l.Dropout(0.5))
    regr_mod.add(l.Dense(1, activation='linear'))

    regr_mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    return regr_mod


def reset_weights(model):
    """
    
    :param model: models for reset weights
    :return: models with initialized weights
    """

    session = backend.get_session()
    for layer in model.layers:
        if isinstance(layer, Container):
            reset_weights(layer)
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)


def cv_split(examples, folds):
    """
    
    :param examples: how many examples in set
    :param folds: how many folds you want
    :return: iterator that yield train and val split
    """
    examples_in_fold = examples // folds
    for i in range(folds):
        val = [j for j in range(examples_in_fold*i, examples_in_fold*(i+1))]
        train = [j for j in range(0, examples_in_fold*i)] + [j for j in range(examples_in_fold*i, examples)]
        yield train, val
