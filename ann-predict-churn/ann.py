# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:55:45 2017

Classification problem
Predict which customers are going to stay as customers

@author: ivan
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import keras libraries
import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential

# Import sklearn libraries 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# -------------------------------
#   Part 1 - Data Preprocessing
# -------------------------------
def preprocess_data(X,y):
    # Encoding categorical variables...
    
    # ... the country
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    
    # ... the gender 
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    # I'm still confused about this part
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]

    # Encoding the Dependent Variable
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    
    return X, y
    
# Importing the dataset
DATASET = 'Churn_Modelling.csv'
dataset = pd.read_csv(DATASET)

# Read data from CreditScore to Estimated salary
X = dataset.iloc[:, 3:13].values

# Read the Ground truth: if a customer exited or not
y = dataset.iloc[:, 13].values

X, y = preprocess_data(X,y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------------
#      Part 2 - Make the ANN
# -------------------------------
def build_classifier(optimizer, use_dropout=False):
    # Initializing the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    # output_dim, number of nodes in the hidden layer. 
    # Rule of thumb = roundup ( inputs/2 )
    classifier.add(Dense(activation="relu", 
                         input_dim=11, 
                         units=6, 
                         kernel_initializer="uniform"))
    if use_dropout:
        classifier.add(Dropout(p = 0.1))
    
    # Add a second hidden layer 
    classifier.add(Dense(activation="relu", 
                         units=6, 
                         kernel_initializer="uniform"))
    if use_dropout:
        classifier.add(Dropout(p = 0.1))
    
    # Add the output layer
    classifier.add(Dense(activation="sigmoid", 
                         units=1, 
                         kernel_initializer="uniform"))
    
    # Compile the ANN (Apply stochastic gradient descent)
    classifier.compile(optimizer = optimizer,
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier

classifier = build_classifier('adam')
# Number of epochs (training on the whole training set)
classifier.fit(X_train,
               y_train,
               batch_size = 10,
               nb_epoch = 100)

# ------------------------------------------------------
#  Part 3 - Making predictions and evaluating the model
# ------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Convert the probabilities into a binary results: threshold
y_pred = (y_pred > 0.5)

# Predicting a single observation
""" Predict a specific customer with the following information
Geo: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
CC: yes
active: yes
estimated salary: 50000 
"""
# encoded country in 2 bits (dummy) variables
# Female is encoded as 0
new_customer_info = [0., 0., 600, 1, 40, 3, 60000, 2, 1, 1, 50000]

# The prediction needs to be in the same scale
new_customer_info = sc.transform(new_customer_info)

# Predicted probability that the customer will leave the bank
new_prediction = classifier.predict(np.array([new_customer_info]))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------------------
#  Part 4 - Evaluating, Improving and Tuning the ANN
# ---------------------------------------------------

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10,
                             nb_epoch = 100)

accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = 1)

mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN (the accuracy was 83%)

# Dropout: solution for overfitting (different accuracy in train and test)
#  or the variance is high in cross-validation
#  Turn off some of the neurons so that the net learns independent relations
#  I added dropout layers into the build classifier method

# Now fine tunning, we'll find the best hyperparameters to increase th accuary
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

# Now get the best results
best_parameters = grid_search.best_params_
best_accuracy =  grid_search.best_score_



















