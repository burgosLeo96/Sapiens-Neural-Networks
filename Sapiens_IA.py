import numpy as np
import keras
import tensorflow
from keras import backend as k 
from keras.models import Sequential
from keras.models import load_model
from flask import Flask
from flask import request
from flask import jsonify
import pickle 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
from collections import OrderedDict
import tensorflow as tf

app = Flask(__name__)



global model
global graph
	






@app.route("/predict/ISIST", methods = ["POST"])
def predictISIST():
    graph = tf.get_default_graph()
    model = load_model('resources/models/ISIST_Model.h5')

    data = {"success": False}
    params = json.load(open('resources/testing/Estudiantes_ISIST.json'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:16]

    with open('resources/trainning/x_train_ISIST.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)
    
    with graph.as_default():
        data["prediction"] = str(model.predict( studentsTable))
        data["success"] = True
        
    #print(StudentsFrame.to_string())
    #response = {'greeting': 'Hello, ' }
    #return jsonify(response)
    k.clear_session()
    return jsonify(data)  


@app.route("/predict/ICVL", methods = ["POST"])
def predictICVL():
    model = load_model('resources/models/ICVL_Model.h5')
    data = {"success": False}
    graph = tf.get_default_graph()
    params = json.load(open('resources/testing/Estudiantes_ICVL.json'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:16]

    with open('resources/trainning/x_train_ICVL.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)
    
    with graph.as_default():
        data["prediction"] = str(model.predict( studentsTable))
        data["success"] = True
        
    #print(StudentsFrame.to_string())
    #response = {'greeting': 'Hello, ' }
    #return jsonify(response)
    k.clear_session()
    return jsonify(data) 

@app.route("/predict/IELEC", methods = ["POST"])
def predictIELEC():
    model = load_model('resources/models/IELEC_Model.h5')
    data = {"success": False}

    graph = tf.get_default_graph() 


    params = json.load(open('resources/testing/Estudiantes_IELEC.json'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:13]

    with open('resources/trainning/x_train_IELEC.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)
    
    with graph.as_default():
        data["prediction"] = str(model.predict( studentsTable))
        data["success"] = True
        
    #print(StudentsFrame.to_string())
    #response = {'greeting': 'Hello, ' }
    #return jsonify(response)
    k.clear_session()
    return jsonify(data) 

@app.route("/predict/IIND", methods = ["POST"])
def predictIIND():
    
    model = load_model('resources/models/IIND_Model.h5')
    data = {"success": False}
    graph = tf.get_default_graph()
    params = json.load(open('resources/testing/Estudiantes_IIND.json'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:16]

    with open('resources/trainning/x_train_IIND.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)
    
    with graph.as_default():
        data["prediction"] = str(model.predict( studentsTable))
        data["success"] = True
        
    #print(StudentsFrame.to_string())
    #response = {'greeting': 'Hello, ' }
    #return jsonify(response)
    k.clear_session()
    return jsonify(data) 


if __name__ == '__main__':
    app.run(debug=True)

