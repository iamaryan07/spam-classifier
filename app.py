import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

# Initialize session state key if it doesn't exist
if 'input_sms' not in st.session_state:
    st.session_state.input_sms = ''

# Callback function to clear input_sms
def clear_text():
    st.session_state.input_sms = ''

# Text area linked to session state
input_sms = st.text_area('Enter your message', key='input_sms')

if st.button('Predict'):
    if input_sms.strip() == '':
        st.warning('Please enter a message to predict.')
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')

# Clear button with callback
st.button('Clear', on_click=clear_text)
