import streamlit as st
import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model_Zyo.h5')
import json
import random

intents = json.loads(open('intentsZyoLarge.json').read())
words = pickle.load(open('texts.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.50
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
        return return_list 
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# STREAMLIT APP
st.markdown('<style>body{text-align:center;background-color:black;color:white;align-items:justify;display:flex;flex-direction:column;}</style>', unsafe_allow_html=True)
st.title("Zyo: Solo Travel Like a Pro!")

st.markdown("Zyo is a chatbot that will guide to explore Indonesia, even if you are alone!")
#print("bot is live")
message = st.text_input("You can ask me anything about Bali, or just share your feelings with me!")
st.markdown('<div style="text-align: justify; font-size: 10pt"><b>Tips:</b> you can ask the FAQ of Bali, like how is the weather, do i need visa, and many more!</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;></div>', unsafe_allow_html=True)
ints = predict_class(message, model)
res = getResponse(ints,intents)
if res != "":
    st.success("Zyo: {}".format(res))
