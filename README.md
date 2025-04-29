# SMS Spam Classifier

## Overview
This project is an **SMS/Email Spam Classifier** built using machine learning. The goal is to classify a given message as either **Spam** or **Not Spam** based on the content of the message. The model is trained using a dataset of labeled messages, and the features are extracted using **TF-IDF** vectorization.

## Project Structure

- `app.py`: The Streamlit app for the SMS spam classifier.
- `vectorizer.pkl`: The pre-trained TF-IDF vectorizer.
- `model.pkl`: The pre-trained machine learning model (e.g., Multinomial Naive Bayes).
- `requirements.txt`: A list of required Python packages.
- `README.md`: Project documentation.

## Setup Instructions

### Prerequisites
1. Python 3.6 or above
2. Streamlit
3. scikit-learn
4. pickle (for loading model and vectorizer)

### Installing Dependencies

1. Clone this repository:

    git clone https://github.com/yourusername/sms-spam-classifier.git
   
2. Navigate to the project directory:

    cd sms-spam-classifier

3. Install the required Python packages using pip:

    pip install -r requirements.txt

### Running the App
To run the application locally, follow these steps:

1. Ensure that vectorizer.pkl and model.pkl files are present in the directory.

2. Run the Streamlit app:

    streamlit run app.py

3. Open your browser and go to the URL provided by Streamlit, typically http://localhost:8501.

## How it Works

-The app takes a message input (SMS or email) from the user.
-The message is preprocessed (lowercased, tokenized, stop words removed, and stemmed).
-The transformed message is then vectorized using the TF-IDF vectorizer.
-The model predicts whether the message is Spam or Not Spam.

# Example Use Cases

Spam Message: "Congratulations! You've won a free ticket to Bahamas!"
Not Spam Message: "Hey, do you have any update on the project?"

# Conclusion
This project demonstrates how to build a machine learning application for spam classification using Streamlit. You can easily deploy this model to a cloud service or use it for real-time classification.
