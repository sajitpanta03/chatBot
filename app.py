from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
from dotenv import load_dotenv
import os
import google.generativeai as genai  # Gemini API

# Initialize environment variables
load_dotenv()  # Load environment variables from .env

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and data
lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize Gemini API
genai.configure(api_key=os.getenv("API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

# Preprocess input text
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Check if user input contains new words
def contains_new_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    for word in sentence_words:
        if word not in words:
            return True  # New word detected
    return False  # No new words

# Predict class for user input
def predict_class(sentence, model, words, classes):
    # Convert sentence to bag of words
    bow_input = bow(sentence, words)
    # Predict the class with the model
    prediction = model.predict(np.array([bow_input]))[0]
    # Get the index with the highest confidence score
    ERROR_THRESHOLD = 0.25
    predicted_class_indices = np.where(prediction > ERROR_THRESHOLD)[0]
    predicted_classes = [classes[i] for i in predicted_class_indices]
    return predicted_classes, prediction

# Get response from Gemini AI
def get_gemini_response(user_input):
    """Get response from Gemini API for financial queries."""
    try:
        response = gemini_model.generate_content(f"You are a helpful assistant. Provide a concise and accurate response to the following query: {user_input}")
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't process your request. Error: {str(e)}"

# Function to get response based on priority
def get_response(intents_json, user_input):
    # Priority 1: Check if the input contains new words
    if contains_new_words(user_input, words):
        return get_gemini_response(user_input)  # Route to AI if new words are detected

    # Priority 2: Check if the intent-based system has a high-confidence match
    predicted_classes, prediction = predict_class(user_input, model, words, classes)
    if predicted_classes:
        for intent in intents_json['intents']:
            if intent['tag'] == predicted_classes[0]:
                # Check if the confidence is high enough
                if max(prediction) > 0.75:  # Confidence threshold
                    if 'responses' in intent and len(intent['responses']) > 0:
                        return random.choice(intent['responses'])

    # Priority 3: Default to AI for all other cases
    return get_gemini_response(user_input)

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    response = get_response(intents, user_input)  # Get response based on priority
    return jsonify(response=response)

if __name__ == "__main__":
    app.run(debug=True)