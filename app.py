import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the stemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem the words
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Streamlit app UI
st.title("Email/SMS Spam Classifier Tool By Yash")

# Input field
input_sms = st.text_area("Enter the message:")

# Prediction logic
if st.button('Predict'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)

    # Vectorize the input
    vector_input = tfidf.transform([transformed_sms])

    # Predict the result
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
