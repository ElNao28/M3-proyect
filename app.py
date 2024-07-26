from flask import Flask, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
from scipy.cluster.hierarchy import fcluster

app = Flask(__name__)

CORS(app)

#Se cargan los modelos ya entrenados
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')
clustersModel = joblib.load('clusters.pkl')

@app.route('/predict', methods=['GET'])
def predict():

    #Se carga el dataSet
    csv_file = 'dataSet-novedades.csv'
    df = pd.read_csv(csv_file)

    #Transformamos la data
    df['total'] = encoder.transform(df[['total']])
    df[['cantidad', 'total']] = scaler.transform(df[['cantidad', 'total']])

    # Se realizan las predicciones de clusters
    df['cluster']  = fcluster(model, t=clustersModel, criterion='maxclust')

    # Identificar el cluster con los valores más altos
    cluster_summary = df.groupby('cluster')[['cantidad', 'total']].mean()
    cluster_summary['suma'] = cluster_summary['cantidad'] + cluster_summary['total']
    clusters = cluster_summary['suma'].idxmax()

    # Obtener id del usuario basado en el  cluster con mayor 'total' y 'cantidad'
    idUsers = df[df['cluster'] == clusters]['userId'].unique()

    response = {
        'idUsers': idUsers.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)