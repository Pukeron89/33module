# <YOUR_IMPORTS>
import glob
import pandas as pd
import dill
import json

from datetime import datetime

def predict():
    # Загрузка обученной модели
    with open('/Users/pukeron89/Downloads/airflow_hw/data/models/cars_pipe_202306101629.pkl', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob('/Users/pukeron89/Downloads/airflow_hw/data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'/Users/pukeron89/Downloads/airflow_hw/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}')

    pass


if __name__ == '__main__':
    predict()
