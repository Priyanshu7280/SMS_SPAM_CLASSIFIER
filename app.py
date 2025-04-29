import streamlit as st
import pickle
import string
import re


# Custom stemming function
def custom_stem(word):
    # Basic stemming by removing common suffixes
    suffixes = ['ing', 'ly', 'ed', 'es', 'er', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:len(word) - len(suffix)]
    return word


# Function to preprocess the text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize using regular expressions (match alphanumeric words)
    words = re.findall(r'\b\w+\b', text)

    # Remove stopwords (you can use a simple set of common stopwords)
    stopwords = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
        'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
        'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
        'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    ])

    # Remove stopwords
    words = [word for word in words if word not in stopwords]

    # Stem words using the custom stem function
    words = [custom_stem(word) for word in words]

    return " ".join(words)


# Load the vectorizer and model (ensure these files exist in the working directory)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)

    # Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])

    # Make a prediction using the loaded model
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
