import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import google.generativeai as genai  # Gemini API
import os
from dotenv import load_dotenv


# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
intents = json.loads(open('intents.json').read())

# Preprocess data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print("Model trained and saved.")

# Load the model (for inference)
model = tf.keras.models.load_model('chatbot_model.h5')

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize Gemini API
load_dotenv()

# Get the API key from the environment variables
GEMINI_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(user_input):
    """Get response from Gemini API for financial queries."""
    try:
        response = gemini_model.generate_content(f"You are a financial assistant. Provide a concise and accurate response to the following query: {user_input}")
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't process your request. Error: {str(e)}"

def predict_intent(user_input):
    """Predict the intent of the user input."""
    bag = [0] * len(words)
    user_words = nltk.word_tokenize(user_input)
    user_words = [lemmatizer.lemmatize(word.lower()) for word in user_words]
    for word in user_words:
        if word in words:
            bag[words.index(word)] = 1
    results = model.predict(np.array([bag]))[0]
    return classes[np.argmax(results)]

def get_response(intent_tag, user_input):
    """Get response based on intent tag."""
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    # If no response found, use Gemini API
    return get_gemini_response(user_input)