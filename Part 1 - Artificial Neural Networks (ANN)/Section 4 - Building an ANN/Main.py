# part 1 : data Preprocessing

import numpy as np
import pandas as pd

#Importing the data set
dataset =  pd.read_csv("Churn_Modelling.csv")
X  = dataset.iloc[:, 3:13].values
Y =  dataset.iloc[:, 13].values

#Encoding  the Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderX1 = LabelEncoder()
X[:, 1] = labelencoderX1.fit_transform(X[:, 1])
labelencoderX2 = LabelEncoder()
X[:,2] = labelencoderX2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X  = onehotencoder.fit_transform(X).toarray()
X = X [ :, 1:]

#Splitting the Data into Training Set and Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)


#Feature Scaling 

from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Now let's Make an ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initiating the ANN
classifier = Sequential()

#Adding the input layer and first Hidden Layer with dropout
classifier .add(Dense(output_dim = 6, init = 'uniform',activation = 'relu', input_dim = 11))
classifier.add(Dropout(p =0.1, ))
#Adding SSecond Hidden Layer
classifier .add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))
classifier.add(Dropout(p =0.1, ))
#Adding the Output Layer
classifier .add(Dense(output_dim = 1, init = 'uniform',activation = 'relu'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,Y_train, batch_size = 10,nb_epoch =100)

#Part 3: Making the predictions and Evaluating the Model
#Predicting the Test set results
Y_pred  = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

#Predicting a SIngle new Observation
'''
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3,60000, 2, 1, 1, 50000 ]])))
new_prediction = (new_prediction >  0.5)
print('The Prediction for the new observation is ', +  new_prediction)

#Part 4: Evaluating, improving and Tuning the AAN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'relu'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10,nb_epoch =100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()
#Improving the ANN
#DropOut Regularization to reduce if needed


#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'relu'))
    classifier.compile(optimizer = optimizer , loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
             'nb_epoch': [100,500],
             'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters, scoring = 'accuracy', cv =10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
