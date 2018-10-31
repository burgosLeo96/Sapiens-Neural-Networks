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

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)



global model
global COMPARATOR
COMPARATOR = 1.0
global PRUEBA_ACADEMICA
PRUEBA_ACADEMICA = 3.3
global MIN_ICFES_MATEMATICAS_IND
MIN_ICFES_MATEMATICAS_IND = 71
global MIN_ICFES_LECTURA_CVL
MIN_ICFES_LECTURA_CVL = 68
global MIN_ICFES_LECTURA_IELEC
MIN_ICFES_LECTURA_IELEC = 65
global MAXIMO_CREDITOS_ACADEMICOS
MAXIMO_CREDITOS_ACADEMICOS = 20
global TASA_APROVACION
TASA_APROBACION = 0.7
global MIN_INSCRIPCION_CREDITOS
MIN_INSCRIPCION_CREDITOS = 32





@app.route("/predict/ISIST", methods = ["POST"])
def predictISIST():
  
    model = load_model('resources/models/ISIST_Model.h5')

    data = {"success": False}
    if request.headers['Content-Type'] != 'application/json':
         return "415 Unsupported Media Type "

    params = json.loads( request.data.decode('utf-8'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:16]
    

    with open('resources/trainning/x_train_ISIST.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)

    StudentsFrame["Prediccion"] = model.predict(studentsTable)

    Response = []


    for i in StudentsFrame.index:
        #Verificar si el estudiante tiene alta probabilidad de estar en estado inactivo
        if StudentsFrame.loc [i,'Prediccion'] < COMPARATOR :
            
            #Desempeño bajo en la carrera
            if float(StudentsFrame.loc[i,'tasa_aprobacion']) < TASA_APROBACION:
                data = { 
                'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Bajo desempeño académico',
                'generador': 'Sistema'
                }
                Response.append(data)
            promedio_primer_semestre = float(StudentsFrame.loc[i,'Promedio Ponderado Acumulado 1'])
            promedio_segundo_semestre = float(StudentsFrame.loc [i,'Promedio Ponderado Acumulado 2']) 
            #Para saber si es de primer semestre académico
            if float(StudentsFrame.loc[i, 'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']) == float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']):
                if promedio_primer_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_primer_semestre > PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Promedio cercano a primera prueba académica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                elif promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Primera prueba academica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                #Algún otro problema no detectado
            #Para estudiantes de segundo semestre académico      
            else:
                #Segunda prueba academica
                if promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    #Verificar si esta en riesgo de caer en segunda prueba 
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a segunda prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Segunda prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                #Primera prueba academica
                else:
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a primera prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Primera prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)

            if not Response:
                #Algún otro problema no detectado
                
                data = { 'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Sin motivo alguno',
                'generador': 'Sistema'
                }
                Response.append(data)
    k.clear_session()
    return json.dumps(Response,ensure_ascii=False).encode('utf8')


@app.route("/predict/ICVL", methods = ["POST"])
def predictICVL():
    model = load_model('resources/models/ICVL_Model.h5')
    data = {"success": False}
    if request.headers['Content-Type'] != 'application/json':
         return "415 Unsupported Media Type "

    params = json.loads( request.data.decode('utf-8'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:16]
    with open('resources/trainning/x_train_ICVL.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)
    
    StudentsFrame["Prediccion"] = model.predict(studentsTable)

    Response = []


    for i in StudentsFrame.index:
        #Verificar si el estudiante tiene alta probabilidad de estar en estado inactivo
        if StudentsFrame.loc [i,'Prediccion'] < COMPARATOR :
            
            #Desempeño bajo en la carrera
            if float(StudentsFrame.loc[i,'tasa_aprobacion']) < TASA_APROBACION:
                data = { 
                'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Bajo desempeño académico',
                'generador': 'Sistema'
                }
                Response.append(data)
            promedio_primer_semestre = float(StudentsFrame.loc[i,'Promedio Ponderado Acumulado 1'])
            promedio_segundo_semestre = float(StudentsFrame.loc [i,'Promedio Ponderado Acumulado 2']) 
            #Para saber si es de primer semestre académico
            if float(StudentsFrame.loc[i, 'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']) == float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']):
                if promedio_primer_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_primer_semestre > PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Promedio cercano a primera prueba académica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                elif promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Primera prueba academica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                # Bajo desempeño prueba icfes lectura critica
                if float(StudentsFrame.loc[i,'Puntaje Lectura Critica']) <= MIN_ICFES_LECTURA_CVL :
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Aptitudes de lectura crítica en prueba saber 11',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
            #Para estudiantes de segundo semestre académico      
            else:
                #Segunda prueba academica
                if promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    #Verificar si esta en riesgo de caer en segunda prueba 
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a segunda prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Segunda prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                #Primera prueba academica
                else:
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a primera prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Primera prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    #Baja inscripción de créditos
                    if float(StudentsFrame.loc[i, 'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']) < MIN_INSCRIPCION_CREDITOS:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Baja inscripción de créditos',
                        'generador': 'Sistema'
                        }
                        Response.append(data)

            if not Response:
                #Algún otro problema no detectado
                
                data = { 'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Sin motivo alguno',
                'generador': 'Sistema'
                }
                Response.append(data)
    

    k.clear_session()
    return json.dumps(Response,ensure_ascii=False).encode('utf8')

@app.route("/predict/IELEC", methods = ["POST"])
def predictIELEC():
    model = load_model('resources/models/IELEC_Model.h5')
    data = {"success": False}
    
    if request.headers['Content-Type'] != 'application/json':
         return "415 Unsupported Media Type "

    params = json.loads( request.data.decode('utf-8'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:13]
    with open('resources/trainning/x_train_IELEC.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)

    StudentsFrame["Prediccion"] = model.predict(studentsTable)

    Response = []


    for i in StudentsFrame.index:
        #Verificar si el estudiante tiene alta probabilidad de estar en estado inactivo
        if StudentsFrame.loc [i,'Prediccion'] < COMPARATOR :
            
            #Desempeño bajo en la carrera
            if float(StudentsFrame.loc[i,'tasa_aprobacion']) < TASA_APROBACION:
                data = { 
                'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Bajo desempeño académico',
                'generador': 'Sistema'
                }
                Response.append(data)
            promedio_primer_semestre = float(StudentsFrame.loc[i,'Promedio Ponderado Acumulado 1'])
            promedio_segundo_semestre = float(StudentsFrame.loc [i,'Promedio Ponderado Acumulado 2']) 
            #Para saber si es de primer semestre académico
            if float(StudentsFrame.loc[i, 'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']) == float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']):
                if promedio_primer_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_primer_semestre > PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Promedio cercano a primera prueba académica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                elif promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Primera prueba academica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                # Bajo desempeño prueba icfes lectura critica
                if float(StudentsFrame.loc[i,'Puntaje Lectura Critica']) <= MIN_ICFES_LECTURA_IELEC :
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Aptitudes de lectura crítica en prueba saber 11',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
            #Para estudiantes de segundo semestre académico      
            else:
                #Segunda prueba academica
                if promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    #Verificar si esta en riesgo de caer en segunda prueba 
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a segunda prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Segunda prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                #Primera prueba academica
                else:
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a primera prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Primera prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    #Baja inscripción de créditos
                    if float(StudentsFrame.loc[i,'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i,'Total Creditos Cursados 1']) < MIN_INSCRIPCION_CREDITOS:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Baja inscripción de créditos',
                        'generador': 'Sistema'
                        }
                        Response.append(data)

            if not Response:
                #Algún otro problema no detectado
                
                data = { 'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Sin motivo alguno',
                'generador': 'Sistema'
                }
                Response.append(data)
    

    k.clear_session()
    return json.dumps(Response,ensure_ascii=False).encode('utf8')


@app.route("/predict/IIND", methods = ["POST"])
def predictIIND():
    
    model = load_model('resources/models/IIND_Model.h5')
    data = {"success": False}
    if request.headers['Content-Type'] != 'application/json':
         return "415 Unsupported Media Type "

    params = json.loads( request.data.decode('utf-8'),object_pairs_hook=OrderedDict)
    StudentsFrame = pd.DataFrame.from_dict(params)
    studentsTable = StudentsFrame.iloc[:,0:16]
    with open('resources/trainning/x_train_IIND.pickle', 'rb') as f:
        x_train = pickle.load(f)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        studentsTable  = sc.transform(studentsTable)

    StudentsFrame["Prediccion"] = model.predict(studentsTable)

    Response = []


    for i in StudentsFrame.index:
        #Verificar si el estudiante tiene alta probabilidad de estar en estado inactivo
        if StudentsFrame.loc [i,'Prediccion'] < COMPARATOR :
            
            #Desempeño bajo en la carrera
            if float(StudentsFrame.loc[i,'tasa_aprobacion'])  < TASA_APROBACION:
                data = { 
                'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Bajo desempeño académico',
                'generador': 'Sistema'
                }
                Response.append(data)
            promedio_primer_semestre = float(StudentsFrame.loc[i,'Promedio Ponderado Acumulado 1'])
            promedio_segundo_semestre = float(StudentsFrame.loc [i,'Promedio Ponderado Acumulado 2']) 
            #Para saber si es de primer semestre académico
            if float(StudentsFrame.loc[i, 'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']) == float(StudentsFrame.loc[i, 'Total Creditos Cursados 1']):
                if promedio_primer_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_primer_semestre > PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Promedio cercano a primera prueba académica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                elif promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Primera prueba academica',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
                # Bajo desempeño prueba icfes matematicas
                if float(StudentsFrame.loc[i,'Puntaje Matematica']) <= MIN_ICFES_MATEMATICAS_IND :
                    data = { 
                    'status': 200,
                    'id': StudentsFrame.loc[i,'ID'],
                    'factor': 'Aptitudes de matemáticas en prueba saber 11',
                    'generador': 'Sistema'
                    }
                    Response.append(data)
            #Para estudiantes de segundo semestre académico      
            else:
                #Segunda prueba academica
                if promedio_primer_semestre <= PRUEBA_ACADEMICA:
                    #Verificar si esta en riesgo de caer en segunda prueba 
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a segunda prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Segunda prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                #Primera prueba academica
                else:
                    if promedio_segundo_semestre  <= ( PRUEBA_ACADEMICA  + 0.2) and  promedio_segundo_semestre > PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Promedio cercano a primera prueba académica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    elif promedio_segundo_semestre <= PRUEBA_ACADEMICA:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Primera prueba academica',
                        'generador': 'Sistema'
                        }
                        Response.append(data)
                    #Baja inscripción de créditos
                    if float(StudentsFrame.loc[i,'Total Creditos Cursados 2']) + float(StudentsFrame.loc[i,'Total Creditos Cursados 1']) < MIN_INSCRIPCION_CREDITOS:
                        data = { 
                        'status': 200,
                        'id': StudentsFrame.loc[i,'ID'],
                        'factor': 'Baja inscripción de créditos',
                        'generador': 'Sistema'
                        }
                        Response.append(data)

            if not Response:
                #Algún otro problema no detectado
                
                data = { 'status': 200,
                'id': StudentsFrame.loc[i,'ID'],
                'factor': 'Sin motivo alguno',
                'generador': 'Sistema'
                }
                Response.append(data)
    

    k.clear_session()
    return json.dumps(Response,ensure_ascii=False).encode('utf8')


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")

