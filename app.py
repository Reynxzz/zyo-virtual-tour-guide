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
import webbrowser

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
            if tag == 'bestressort_faq':
                followup_bestresorts()
                result = ""
                break
            elif tag == 'besthotel_faq':
                followup_besthotels()
                result = ""
                break
            elif tag == 'just_talk':
                generative_model()
            elif tag == 'recomend_destination':
                recommend_destination()
                result = ""
                break
            result = random.choice(i['responses'])
    return result
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def followup_bestresorts():
    st.success("Zyo: Bali has many beautiful resorts to stay, i will suggest you the popular one, AYANA Resorts & Spa Bali!")
    url = 'https://www.google.com/travel/hotels/Bali/entity/CgoIyo2FjenOuZ8xEAE?q=best%20hotel%20in%20bali&g2lb=2502548%2C2503771%2C2503781%2C4258168%2C4270442%2C4284970%2C4291517%2C4306835%2C4597339%2C4703207%2C4718358%2C4723331%2C4757164%2C4790928%2C4809518%2C4812117%2C4814050%2C4816977%2C4826688%2C4828448%2C4829505%2C4845220&hl=id-ID&gl=id&cs=1&ssta=1&ts=CAESABogCgIaABIaEhQKBwjmDxAJGBwSBwjmDxAJGB0YATICEAAqCwoFOgNJRFIaACgI&rp=EMqNhY3pzrmfMRCc25_c1ezGtkgQ3ZDq-L2xwYEbEOC3lOzt7KH_pAE4AUAASAKiAQRCYWxpwAED&ap=aAE&ictx=1&ved=0CAAQ5JsGahcKEwiA3ZLykfr5AhUAAAAAHQAAAAAQBA&utm_campaign=sharing&utm_medium=link&utm_source=htls'
    st.warning("I can redirect you to Google Maps for the best experience")
    st.image("https://lh3.googleusercontent.com/p/AF1QipP7g50r08kUTJ0vd_TtIrIyPzEYYyShFRK5D02w=w592-h404-n-k-rw-no-v1")
    if st.button("Let's Go!"):
        webbrowser.open_new_tab(url)

def followup_besthotels():
    st.success("Zyo: Bali has many beautiful hotels to stay, i will suggest you the popular one, Double-Six Luxury Hotel!")
    url = 'https://www.google.com/travel/hotels/Bali/entity/CgsI4LeU7O3sof-kARAB/overview?q=best%20hotel%20in%20bali&g2lb=2502548%2C2503771%2C2503781%2C4258168%2C4270442%2C4284970%2C4291517%2C4306835%2C4597339%2C4703207%2C4718358%2C4723331%2C4757164%2C4790928%2C4809518%2C4812117%2C4814050%2C4816977%2C4826688%2C4828448%2C4829505%2C4845220&hl=id-ID&gl=id&cs=1&ssta=1&ts=CAESABogCgIaABIaEhQKBwjmDxAJGBwSBwjmDxAJGB0YATICEAAqCwoFOgNJRFIaACgI&rp=EMqNhY3pzrmfMRCc25_c1ezGtkgQ3ZDq-L2xwYEbEOC3lOzt7KH_pAE4AUAASAKiAQRCYWxpwAED&ap=aAE&ictx=1&utm_campaign=sharing&utm_medium=link&utm_source=htls'
    st.warning("I can redirect you to Google Maps for the best experience")
    st.image("https://lh3.googleusercontent.com/p/AF1QipOcOYwjvpsX2emDQvCYC3lclDKCGbzaPWFTG400=w592-h404-n-k-rw-no-v1", width=700)
    if st.button("Let's Go!"):
        webbrowser.open_new_tab(url)

def recommend_destination():
    st.success("Bali is bursting with amazing places to visit and explore, Whether exploring the distinctive culture of the Balinese people, scuba diving in coral reefs, climbing an ancient volcano or sunbathing on a broad stretch of beach, Bali has a bit of paradise to offer every visitor")
    st.warning("What kind of place you want to explore?")
    place = st.radio( "Select one!",
     ('Beach', 'Cultural Place', 'Entertainment'))
    
    if place == 'Beach':
        st.write('Bali Has many beautiful beaches!')
        st.write('if you are a beginner in Bali, i suggest you to go to Kuta Beach, The most popular beach in Bali!')
        st.image('https://indonesiakaya.com/wp-content/uploads/2020/10/pantai_kuta_bali_1200.jpg')
        st.warning('i can redirect you to a good article for this topic, you can press the button below!')
        if st.button("Another beach, please!"):
            url = "https://thehoneycombers.com/bali/best-beach-bali-swim-surf-sand/"
            webbrowser.open_new_tab(url)

    elif place == 'Cultural Place':
        st.write("Beyond the island's clear aesthetic beauty though, I'd say some of Bali's most incredible wonders are its cultural ones!")
        st.write("Ubud Palace is one of the top cultural must-sees here in Bali's spiritual epicentre.")
        st.image('https://happytowander.com/wp-content/uploads/Ubud-Bali-Indonesia-Trip-of-Wonders-Happy-to-Wander-5360.jpg')
        st.warning('i can redirect you to a good article for this topic, you can press the button below!')
        if st.button("Another places, please!"):
            url = "https://happytowander.com/bali-culture-bucket-list-8-awesome-cultural-things-to-do-in-bali/"
            webbrowser.open_new_tab(url)

    elif place == 'Entertainment':
        st.write("What's a trip to Bali if you don't hit at least two or three nightclubs during your stay?")
        st.write("Opened 7 days a week, La Favela is a must-go if you're new to Bali's nightlife scene")
        st.image('https://uncoverasia.com/wp-content/uploads/2019/10/La-Favela.png')
        st.warning('i can redirect you to a good article for this topic, you can press the button below!')
        if st.button("Another club, please!"):
            url = "https://uncoverasia.com/best-clubs-in-bali/"
            webbrowser.open_new_tab(url)

# GENERATIVE MODEL CHATBOT
# --------------------------------------------------------------- #
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("lizz27/DialoGPT-medium-BaymaxBot")
 model = AutoModelForCausalLM.from_pretrained("lizz27/DialoGPT-medium-BaymaxBot")
 return tokenizer, model

def generative_model():
    tokenizer, model = load_data()

    st.write("###### You can talk to me anything you want. I hope i can understand you :)")
    input = st.text_input('Talk to me anything:')
    if 'count' not in st.session_state or st.session_state.count == 6:
        st.session_state.count = 0 
        st.session_state.chat_history_ids = None
        st.session_state.old_response = ''
    else:
        st.session_state.count += 1

    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if st.session_state.old_response == response:
        bot_input_ids = new_user_input_ids
    
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.write(f"Zyo: {response}")

    st.session_state.old_response = response

# --------------------------------------------------------------- #



# STREAMLIT APP
st.set_page_config(page_title="Zyo", page_icon="ZyoLogo.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.markdown('<style>body{text-align:center;background-color:black;color:white;align-items:justify;display:flex;flex-direction:column;}</style>', unsafe_allow_html=True)
st.title("Zyo: Solo Travel Like a Pro!")

st.markdown("Zyo is a chatbot that will guide you to explore Indonesia, even if you are alone!")
st.image("ZyoLanding.png")
#print("bot is live")
message = st.text_input("You can ask me anything about Bali, or just share your feelings with me!")
st.markdown('<div style="text-align: justify; font-size: 10pt"><b>Tips:</b> you can ask the FAQ of Bali, like how is the weather, do i need visa, and many more!</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;></div>', unsafe_allow_html=True)
ints = predict_class(message, model)
res = getResponse(ints,intents)
if res != "":
    st.success("Zyo: {}".format(res))
