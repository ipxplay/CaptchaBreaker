from time import time

import numpy as np
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer

from iitg import config
from iitg.core.train_model import read_data_labels


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def lenet5_model(filters1=32, k_size1=3, filters2=32, k_size2=3,
                 act='relu', lr=0.01, decay=0, dropout=0):
    model = Sequential()
    inputShape = (config.INPUT_SIZE, config.INPUT_SIZE, 1)

    # first set of CONV=>RELU=>POOL layers
    model.add(Conv2D(filters1, (k_size1, k_size1),
                     input_shape=inputShape, padding='same'))
    model.add(Activation(act))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))

    # second set of CONV=>RELU=>POOL layers
    model.add(Conv2D(filters2, (k_size2, k_size2), padding='same'))
    model.add(Activation(act))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(act))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(28))
    model.add(Activation('softmax'))

    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)
    # opt = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = KerasClassifier(build_fn=lenet5_model)

    filters1 = [6, 16, 32, 64]
    k_size1 = [3, 5]
    filters2 = [6, 16, 32, 64]
    k_size2 = [3, 5]
    act = ['relu', 'elu']
    lr = [1e-2, 1e-3, 1e-4]
    decay = [1e-6, 1e-9, 0]
    dropout = [0, 0.1, 0.2, 0.3]

    # 测试速度
    # filters1 = [6]
    # k_size1 = [3]
    # filters2 = [6]
    # k_size2 = [3]
    # act = ['relu']
    # lr = [1e-2]
    # decay = [1e-6]
    # dropout = [0.2]

    hyperparams = {
        'filters1': filters1,
        'k_size1': k_size1,
        'filters2': filters2,
        'k_size2': k_size2,
        'act': act,
        'lr': lr,
        'decay': decay,
        'dropout': dropout
    }
    data, labels = read_data_labels()
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)
    lb = LabelBinarizer().fit(labels)
    labels = lb.transform(labels)

    n_iter_search = 8
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=hyperparams,
                                       n_iter=n_iter_search,
                                       n_jobs=-2,
                                       cv=4,
                                       verbose=2,
                                       random_state=42)

    start = time()
    random_search.fit(data, labels,
                      epochs=20, batch_size=64, verbose=2)

    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)
