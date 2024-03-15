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
from joblib import load, Parallel, delayed
app = Flask(__name__)
#CORS(app)

intents = {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "How are you?"],
      "responses": [
        "Hello! How can I assist you?", 
        "Hi there! How can I help you today?", 
        "Hey! What can I do for you?", 
        "Hi! Welcome to IntelliChat. How may I help?", 
        "Hello! Nice to meet you. How may I assist?"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "Goodbye", "See you later"],
      "responses": [
        "Goodbye! Have a great day.", 
        "See you later!", 
        "Farewell! Take care.", 
        "Bye bye! Come back soon.", 
        "Goodbye! Don't hesitate to return if you need assistance."
      ]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "Appreciate it"],
      "responses": [
        "You're welcome!", 
        "No problem, happy to help!", 
        "Anytime!", 
        "Glad I could assist!", 
        "You're most welcome!"
      ]
    },
    {
      "tag": "age",
      "patterns": ["How old are you?", "What's your age?"],
      "responses": [
        "I'm just a computer program, so I don't have an age.", 
        "Age is just a number for me!", 
        "I don't age, I just get updated.", 
        "I'm timeless!", 
        "I'm ageless, like a timeless piece of software."
      ]
    },
    {
      "tag": "weather",
      "patterns": ["What's the weather like?", "How's the weather today?"],
      "responses": [
        "You can check the weather forecast online or on your phone.", 
        "I'm not a meteorologist, but you can find weather updates online.", 
        "I'm afraid I can't provide real-time weather updates. Try a weather app!", 
        "I don't have a weather forecast feature, but I can help with other queries.", 
        "Weather conditions vary by location. You might want to check a weather website."
      ]
    },
    {
      "tag": "joke",
      "patterns": ["Tell me a joke", "Do you know any jokes?"],
      "responses": [
        "Why don't scientists trust atoms? Because they make up everything!", 
        "What do you call fake spaghetti? An impasta!", 
        "Why did the scarecrow win an award? Because he was outstanding in his field!", 
        "Why couldn't the bicycle stand up by itself? It was two-tired!", 
        "What's orange and sounds like a parrot? A carrot!"
      ]
    },
    {
      "tag": "identity",
      "patterns": ["Who are you?", "What are you?", "Tell me about yourself", "What's your purpose?", "Who created you?"],
      "responses": [
        "I'm IntelliChat, a virtual assistant designed to help you with your questions and tasks.", 
        "I'm IntelliChat, an AI-powered chatbot programmed to provide assistance and information.", 
        "I'm IntelliChat, your friendly neighborhood chatbot here to assist you!", 
        "I'm IntelliChat, a digital assistant created to make your life easier.", 
        "I'm IntelliChat, a conversational agent designed to engage with users and provide helpful responses."
      ]
    },
    {
      "tag": "technology",
      "patterns": ["What technology are you built with?", "How do you work?", "What's your underlying technology?", "What programming language are you written in?"],
      "responses": [
        "I'm built using natural language processing (NLP) techniques and machine learning algorithms.", 
        "I use artificial intelligence (AI) to understand and respond to user queries.", 
        "My underlying technology includes deep learning models and neural networks.", 
        "I'm programmed using Python and utilize libraries like TensorFlow and NLTK.", 
        "I leverage advanced algorithms to analyze text data and generate responses."
      ]
    },
    {
      "tag": "capabilities",
      "patterns": ["What can you do?", "What are your abilities?", "What tasks can you perform?", "How can you assist me?"],
      "responses": [
        "I can provide information on a wide range of topics, answer questions, and assist with tasks.", 
        "I'm capable of understanding natural language queries and providing relevant responses.", 
        "I can help you with tasks such as finding information, making recommendations, and providing support.", 
        "I'm designed to assist users by providing accurate and helpful responses to their queries.", 
        "My capabilities include answering questions, providing recommendations, and offering support."
      ]
    }
    
  ]
}

# Load preprocessed data
words = load('words.pkl')
classes = load('classes.pkl')
nb_classifier = load('nb_classifier.joblib')

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
