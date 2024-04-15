from flask import Flask, request, jsonify
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import numpy as np

# Ensure you have these NLTK datasets downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model = joblib.load('random_forest_regressor.joblib')
# Load the saved TF-IDF Vectorizer
vectorizer = joblib.load('tfidf_vectorizer.joblib')

app = Flask(__name__)

def preprocess_text(text):
    # Placeholder for your actual preprocessing functions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove all non-alphabet characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [token for token in tokens if token.isalpha()]  # Keep only alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return ' '.join(tokens)  # Return processed text as a single string

@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, World! I am ready to help!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review_text = data.get('review', '')  # Assuming JSON has a 'review' key
    processed_text = preprocess_text(review_text)

    # Transform the processed text using the loaded vectorizer
    transformed_text = vectorizer.transform([processed_text])
    print('transformed_text', transformed_text)
    
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = np.argsort(transformed_text.toarray()).flatten()[::-1]

    # Get top 3 significant words
    top_words = [feature_array[idx] for idx in tfidf_sorting[:3]]

    # Assuming your model expects an array of inputs
    prediction = model.predict(transformed_text)
    prediction = [round(num, 2) for num in prediction]  # Rounds to two decimal places
    print('Prediction:', prediction)

    # Return a detailed JSON response
    return jsonify({
        'prediction': prediction,
        'top_influential_words': top_words
    })

if __name__ == '__main__':
    app.run(debug=True)