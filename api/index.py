from flask import Flask, request, jsonify, render_template
#from flask_cors import CORS
import json
import random
import wikipedia
import re
import time
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import joblib

app = Flask(__name__)
#CORS(app)

# Load intents data from JSON file
intents = json.loads(open('intents.json').read())

# Load preprocessed data
words = joblib.load('words.pkl')
classes = joblib.load('classes.pkl')
nb_classifier = joblib.load('nb_classifier.joblib')

lemmatizer = WordNetLemmatizer()

# Function to clean up a sentence by tokenizing and lemmatizing its words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert a sentence into bag of words representation
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if lemmatizer.lemmatize(word.lower()) in sentence_words else 0 for word in words]
    return np.array(bag)

# Function to predict the intent class of a given sentence
def predict_class(sentence):
    p = bow(sentence, words)
    res = nb_classifier.predict(np.array([p]))[0]
    return_list = [{"intent": classes[res], "probability": "1"}]
    return return_list

# Function to get a response based on predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to extract subject from a question
def extract_subject(question):
    punctuation_marks = ['.', ',', '!', '?', ':', ';', "'", '"', '(', ')', '[', ']', '-', '—', '...', '/', '\\', '&', '*', '%', '$', '#', '@', '+', '-', '=', '<', '>', '_', '|', '~', '^']
    for punctuation_mark in punctuation_marks:
        if punctuation_mark in question:
            question = question.replace(punctuation_mark, '')
    
    subject = ''
    words = question.split(' ')
    list_size = len(words)

    for i in range(list_size):
        if i > 1 and i != list_size:
            subject += words[i]+' '
        elif i == list_size:
            subject += words[i]
    return subject

# Function to clean text by removing characters within parentheses
def clean_text(text):
    cleaned_text = re.sub(r'\([^()]*\)', '', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to search Wikipedia for information based on a question
def search_wikipedia(question, num_sentences=2):
    try:
        subject = extract_subject(question)
        wiki_result = wikipedia.summary(subject, auto_suggest=False, sentences=num_sentences)
        return clean_text(wiki_result)
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find information about {subject}."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple matches found. Try being more specific: {', '.join(e.options)}"
    except Exception as e:
        return "Error, Something went wrong!"

# Function to get a response from the chatbot
def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res
    
@app.route('/chat', methods=['POST'])
def chat():
    user_text = request.form['user_input']
    bot_response = chatbot_response(user_text)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run()
