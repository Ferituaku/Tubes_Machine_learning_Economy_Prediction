from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import logging

# Load scaler dan model
scaler = pickle.load(open('scaler.pkl', 'rb'))
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Route utama untuk menampilkan form input
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) == 0:
            return jsonify({'error': 'Input features tidak boleh kosong.'})

        features_scaled = scaler.transform([features])
        cluster = kmeans_model.predict(features_scaled)[0]

        cluster_descriptions = {
            0: "Negara berkembang (moderate scores).",
            1: "Negara tertinggal (low scores).",
            2: "Negara maju (high safety, governance, dan health).",
        }

        return jsonify({
            'cluster': int(cluster),
            'description': cluster_descriptions[cluster]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
