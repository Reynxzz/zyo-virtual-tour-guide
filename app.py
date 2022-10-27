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
import streamlit_authenticator as stauth
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit.components.v1 as components

import database as db

st.set_page_config(page_title="Zyo", page_icon="ZyoLogo.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

# Recommender System - Hotels
@st.cache(allow_output_mutation=True)
def load_data_hotel():
    df = pd.read_excel("datahotelZyo.xlsx")
    df['types'] = df.types.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("types")
    return exploded_track_df

tipe_names = ['Hotel', 'Resort']
map_feats = ["gymSwimmingPoolFacility", "affordable", "restaurant", "beachView", "recreation", "soloFriendly"]

exploded_track_df = load_data_hotel()

def n_neighbors_url_map(tipe, test_feat):
    tipe = tipe.lower()
    tipe_data = exploded_track_df[(exploded_track_df["types"]==tipe)]
    tipe_data = tipe_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(tipe_data[map_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(tipe_data), return_distance=False)[0]

    urls = tipe_data.iloc[n_neighbors]["url"].tolist()
    maps = tipe_data.iloc[n_neighbors][map_feats].to_numpy()
    return urls, maps

def recommend_hotels():
    st.write("What kind of places do you want me to recommend you?")
    tipe = st.radio(
        "",
    tipe_names, index=tipe_names.index("Hotel"))
    st.markdown("")

    st.write("What kind of facilities do you like? (slide this based on your preferences)")
    with st.container():
        col1, col2,= st.columns((2,2))
        with col2:
            beach = st.slider(
                'Beach View',
                0, 100, 50)
            recreation = st.slider(
                'Recreation Place',
                0, 100, 50)
            solofriendly = st.slider(
                'Solo-friendly',
                0, 100, 50)
        with col1:
            gym = st.slider(
                'Gym & Swimming Pool Facility',
                0, 100, 50)
            affordable = st.slider(
                'Affordable',
                0, 100, 50)
            restaurant = st.slider(
                'Restaurant',
                0, 100, 50)
            

    tracks_per_page = 6
    test_feat = [gym, affordable, restaurant, beach, recreation, solofriendly]
    urls, maps = n_neighbors_url_map(tipe, test_feat)

    tracks = []
    for url in urls:
        track = """<div class="mapouter"><div class="gmap_canvas"><iframe class="gmap_iframe" width="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="https://maps.google.com/maps?width=400&amp;height=400&amp;hl=en&amp;q={}&amp;t=&amp;z=14&amp;ie=UTF8&amp;iwloc=B&amp;output=embed"></iframe><h4 style="color:white;font-family:Arial">{}</h4></div><style>.mapouter.gmap_canvas.gmap_iframe </style></div>""".format(url,url)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [tipe] + test_feat
    
    current_inputs = [tipe] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    st.write("This is the best place for you!")
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Hotels"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_maps = maps[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, map) in enumerate(zip(current_tracks, current_maps)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=200,
                        )
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=200,
                        )
        else:
            st.write("No places left to recommend")

# ---------------------------------------------------------------- #

# Recommender System - Destinations
@st.cache(allow_output_mutation=True)
def load_data_destination():
    df = pd.read_excel("datadestinationZyo.xlsx")
    df['types'] = df.types.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("types")
    return exploded_track_df

tipe_names = ['cultural', 'nature']
map_feats = ["affordable", "exotic", "unique", "culture","transportation", "soloFriendly"]

exploded_track_df = load_data_destination()

def recommend_destination():
    st.write("What kind of places do you want me to recommend you?")
    tipe = st.radio(
        "",
    tipe_names, index=tipe_names.index("cultural"))
    st.markdown("")

    st.write("What kind of facilities do you like? (slide this based on your preferences)")
    with st.container():
        col1, col2,= st.columns((2,2))
        with col2:
            culture = st.slider(
                'Culture',
                0, 100, 50)
            transportation = st.slider(
                'Transportation',
                0, 100, 50)
            solofriendly = st.slider(
                'Solo-friendly',
                0, 100, 50)
        with col1:
            affordable = st.slider(
                'Affordable',
                0, 100, 50)
            exotic = st.slider(
                'Exotic',
                0, 100, 50)
            unique = st.slider(
                'Unique',
                0, 100, 50)
            

    tracks_per_page = 6
    test_feat = [affordable, exotic, unique, culture, transportation, solofriendly]
    urls, maps = n_neighbors_url_map(tipe, test_feat)

    tracks = []
    for url in urls:
        track = """<div class="mapouter"><div class="gmap_canvas"><iframe class="gmap_iframe" width="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="https://maps.google.com/maps?width=400&amp;height=400&amp;hl=en&amp;q={}&amp;t=&amp;z=14&amp;ie=UTF8&amp;iwloc=B&amp;output=embed"></iframe><h4 style="color:white;font-family:Arial">{}</h4></div><style>.mapouter.gmap_canvas.gmap_iframe </style></div>""".format(url,url)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [tipe] + test_feat
    
    current_inputs = [tipe] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    st.write("This is the best place for you!")
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Destinations"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_maps = maps[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, map) in enumerate(zip(current_tracks, current_maps)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=200,
                        )
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=200,
                        )
        else:
            st.write("No places left to recommend")

# ---------------------------------------------------------------- #




# DIALOGPT CHATBOT
# --------------------------------------------------------------- #
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 st.warning("Wait a second, i have to prepare myself to talk...")
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
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
    
    st.session_state.chat_history_ids.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, temperature=0.9, repetition_penalty=1.5)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.success(f"Zyo: {response}")

    st.session_state.old_response = response

# --------------------------------------------------------------- #


# --- USER AUTHENTICATION ---
users = db.fetch_all_users()

usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef")

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
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
                    recommend_hotels()
                    result = ""
                    break
                elif tag == 'besthotel_faq':
                    recommend_hotels()
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

    
    # STREAMLIT APP
    st.markdown('<style>body{text-align:center;background-color:black;color:white;align-items:justify;display:flex;flex-direction:column;}</style>', unsafe_allow_html=True)
    st.title("Zyo: Solo Travel Like a Pro!")

    st.markdown("Zyo is a chatbot that will guide you to explore Indonesia, even if you are alone!")
    st.image("ZyoLanding.png")
    message = st.text_input("You can ask me anything about Bali, or just share your feelings with me!")
    st.markdown('<div style="text-align: justify; font-size: 10pt"><b>Tips:</b> you can ask the FAQ of Bali, like how is the weather, do i need visa, and many more!</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;></div>', unsafe_allow_html=True)
    ints = predict_class(message, model)
    res = getResponse(ints,intents)
    if res != "":
        st.success("Zyo: {}".format(res))
    
    authenticator.logout("Logout", "main")