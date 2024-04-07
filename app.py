from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('random_forest_regressor.joblib')

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the data
    processed_data = preprocess_data(data['review'])
    # Predict with your model
    prediction = model.predict(processed_data)
    return jsonify(prediction=prediction[0])

def preprocess_data(review):
    # Perform the same preprocessing as during training...
    pass

if __name__ == '__main__':
    app.run(debug=True)