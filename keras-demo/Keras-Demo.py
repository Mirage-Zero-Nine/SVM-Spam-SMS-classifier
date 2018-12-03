__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: Keras-Demo
Create Time: 2018-11-25 14:52
'''

import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pandas

'''
Install TensorFlow in Python 3.7:
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl

**USE PYTHON 3.6 TO INSTALL TENSORFLOW!!!**
**DO NOT USE PYTHON 3.7!!!**
'''


def keras_demo(dataset=numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")):
    # Fix random seed
    # seed = 7
    # numpy.random.seed(seed)

    # Load dataset

    # Split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    '''
    Define model with each layer.
    input_dim: input number
    init: initialize method, default method is uniform distribution
    activation: activation function
    '''
    model = Sequential()
    model.add(Dense(20, input_dim=8, activation='relu', kernel_initializer='uniform'))  # Hidden layer, 12 neurons
    model.add(Dense(8, activation='relu', kernel_initializer='uniform'))  # Hidden layer, 8 neurons
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))  # Output layer

    # Training model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fitting model
    model.fit(X, Y, nb_epoch=500, batch_size=50)

    # Evaluate the model
    scores = model.evaluate(X, Y)

    print('====================')
    print("Average Accuracy: %.2f%%" % (scores[1] * 100))
    print('====================')


def kears_demo_with_auto_check(dataset=numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")):
    start = datetime.datetime.now()

    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
        model.add(Dense(8, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # create model
    model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

    end = datetime.datetime.now()
    time = end - start
    print("Consume time: " + str(time))


def kears_demo_with_k_fold(dataset=numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")):
    start = datetime.datetime.now()
    # MLP for Pima Indians Dataset with 10-fold cross validation
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # define 10-fold cross validation test harness
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    for train_index, test_index in skf.split(X, Y):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(8, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Fit the model

        model.fit(X[train_index], Y[train_index], epochs=150, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test_index], Y[test_index], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    print("Test complete.")

    end = datetime.datetime.now()
    time = end - start
    print("Consume time: " + str(time))


def kears_demo_with_grid_search():
    start = datetime.datetime.now()

    # MLP for Pima Indians Dataset with grid search via sklearn

    # Function to create model, required for KerasClassifier
    def create_model(optimizer='rmsprop', init='glorot_uniform'):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu', kernel_initializer=init, ))
        model.add(Dense(8, activation='relu', kernel_initializer=init, ))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=init, ))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # create model
    model = KerasClassifier(build_fn=create_model)
    # grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = numpy.array([50, 100, 150])
    batches = numpy.array([5, 10, 20])
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)
    end = datetime.datetime.now()
    time = end - start
    print("Consume time: " + str(time))

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.cv_results_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


if __name__ == '__main__':
    kears_demo_with_k_fold()

    kears_demo_with_grid_search()
