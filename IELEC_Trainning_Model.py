# import panda, keras and tensorflow
import pandas as pd
import tensorflow as tf
import keras
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
import pickle


# Load the sample data set and split into x and y data frames 
dataset = pd.read_csv("resources\data\entradaIELEC.csv", sep=';', encoding='latin-1')
x = dataset.iloc[:,0:13].values;
y = dataset.iloc[:,13].values;
#Splitting the dataset into the Training set and Test set
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state =0 )
with open('resources/trainning/x_train_IELEC.pickle', 'wb') as f:
    pickle.dump(x_train,f)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#Initialising the ANN
classifier = Sequential()

# Adding the input layer and  the first hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu',input_dim = 13))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
  
  
# Fitting the ANN to the Training set
classifier.fit(x_train,y_train,batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
#y_pred = classifier.predict(x_test)

results = classifier.evaluate(x_test,y_test, verbose= 0)
print(results)

classifier.save("resources\models\IELEC_Model.h5")

