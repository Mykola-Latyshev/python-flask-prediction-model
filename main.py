import numpy as np
import pandas as pd

import pickle
import xgboost as xgb
from xgboost import XGBClassifier

from flask_restful import Resource, Api
from flask_cors import CORS
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/status', methods=['GET'])
def a_live():
    return "Alive!"


@app.route('/api_03', methods=['POST'])
def predict():
    data = request.get_json()

    df_data = pd.json_normalize(data)


    model = pickle.load(open('C:\model\model_03_all.pkl', 'rb'))
    prediction = model.predict_proba(df_data)

    df_proba = pd.DataFrame(prediction.tolist()[0], columns=['proba'],
                            index=['psoriasis', 'seboreic_dermatitis', 'lichen_planus', 'pityriasis_rosea',
                                   'cronic_dermatitis', 'pityriasis_rubra_pilaris'])
    df_sort_proba = df_proba.sort_values('proba', ascending=False)
    df_sort_proba['Class'] = df_sort_proba.index

    result = {'main_class_output': str(df_sort_proba['Class'].iloc[0]),
              'main_proba': str(df_sort_proba['proba'].iloc[0]),
              'second_class_output': str(df_sort_proba['Class'].iloc[1]),
              'second_proba': str(df_sort_proba['proba'].iloc[1]),
              'third_class_output': str(df_sort_proba['Class'].iloc[2]),
              'third_proba': str(df_sort_proba['proba'].iloc[2])
              }
    return result



@app.route('/api_02', methods=['POST'])
def predict_02():
    data_02 = request.get_json()

    df_data_02 = pd.json_normalize(data_02)

    model_02 = pickle.load(open('C:\model\model_02_all.pkl', 'rb'))
    prediction = model_02.predict_proba(df_data_02)

    df_proba_02 = pd.DataFrame(prediction.tolist()[0], columns=['proba'],
                            index=['psoriasis', 'seboreic_dermatitis', 'lichen_planus', 'pityriasis_rosea',
                                   'cronic_dermatitis', 'pityriasis_rubra_pilaris'])
    df_sort_proba_02 = df_proba_02.sort_values('proba', ascending=False)
    df_sort_proba_02['Class'] = df_sort_proba_02.index

    result_02 = {'main_class_output': str(df_sort_proba_02['Class'].iloc[0]),
              'main_proba': str(df_sort_proba_02['proba'].iloc[0]),
              'second_class_output': str(df_sort_proba_02['Class'].iloc[1]),
              'second_proba': str(df_sort_proba_02['proba'].iloc[1]),
              'third_class_output': str(df_sort_proba_02['Class'].iloc[2]),
              'third_proba': str(df_sort_proba_02['proba'].iloc[2])
              }
    return result_02


@app.route('/api_01', methods=['POST'])
def predict_01():
    data_01 = request.get_json()

    df_data_01 = pd.json_normalize(data_01)

    model_01 = pickle.load(open('C:\model\model_01_all.pkl', 'rb'))
    prediction_01 = model_01.predict_proba(df_data_01)

    df_proba_01 = pd.DataFrame(prediction_01.tolist()[0], columns=['proba'],
                            index=['psoriasis', 'seboreic_dermatitis', 'lichen_planus', 'pityriasis_rosea',
                                   'cronic_dermatitis', 'pityriasis_rubra_pilaris'])
    df_sort_proba_01 = df_proba_01.sort_values('proba', ascending=False)
    df_sort_proba_01['Class'] = df_sort_proba_01.index

    result_01 = {'main_class_output': str(df_sort_proba_01['Class'].iloc[0]),
              'main_proba': str(df_sort_proba_01['proba'].iloc[0]),
              'second_class_output': str(df_sort_proba_01['Class'].iloc[1]),
              'second_proba': str(df_sort_proba_01['proba'].iloc[1]),
              'third_class_output': str(df_sort_proba_01['Class'].iloc[2]),
              'third_proba': str(df_sort_proba_01['proba'].iloc[2])
              }
    return result_01


if __name__ == '__main__':
    app.run(port=5000)