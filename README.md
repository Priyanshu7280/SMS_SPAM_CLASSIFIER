# SMS Spam Classifier

## Overview
This project is an **SMS/Email Spam Classifier** built using machine learning. The goal is to classify a given message as either **Spam** or **Not Spam** based on the content of the message. The model is trained using a dataset of labeled messages, and the features are extracted using **TF-IDF** vectorization.

## Project Structure

- `app.py`: The Streamlit app for the SMS spam classifier.
- `rain_and_save.py`: Script to retrain and save the model/vectorizer
- `check_pickle.py`: A utility script to check the validity of the pickle files (model/vectorizer).
- `spam.csv`: The dataset for training the model (not required for running the app).
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

---

### Installing Dependencies

1. **Clone the repository**

    Open a terminal and run the following command to clone the repository:

    ```bash
    git clone https://github.com/yourusername/sms-spam-classifier.git
    ```

2. **Navigate to the project directory**

    Change your directory to the project folder:

    ```bash
    cd sms-spam-classifier
    ```

3. **Install the required Python packages**

    Run the following command to install the dependencies listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

---

### Running the App

To run the application locally, follow these steps:

1. Ensure that the `model.pkl` and `vectorizer.pkl` files are present in the `models/` directory.

2. **Run the Streamlit app**

    Run the following command to start the app:

    ```bash
    streamlit run app.py
    ```

3. **Access the app**

    After the app starts, open your browser and go to the URL provided by Streamlit, which is usually:

    ```
    http://localhost:8501
    ```

---

### Retraining the Model

To retrain the model with new data, use the `train_and_save.py` script. 

1. Place your dataset (`spam.csv`) inside the `data/` folder.
2. Run the following command:

    ```bash
    python train_and_save.py
    ```

This will regenerate:
- `models/model.pkl` — the trained model file.
- `models/vectorizer.pkl` — the trained vectorizer file.

---

### Check Pickle Validity

If you want to verify that your `model.pkl` and `vectorizer.pkl` files are correctly saved and can be loaded, use the `check_pickle.py` utility.

To check the pickle files, run:

```bash
python check_pickle.py
``
---

## How it Works

1. The app takes a message input (SMS or email) from the user.
2. The message is preprocessed (lowercased, tokenized, stop words removed, and stemmed).
3. The transformed message is then vectorized using the TF-IDF vectorizer.
4. The model predicts whether the message is Spam or Not Spam.

## Example Use Cases

1. Spam Message: "Congratulations! You've won a free ticket to Bahamas!"
2. Not Spam Message: "Hey, do you have any update on the project?"

# Conclusion
This project demonstrates how to build a machine learning application for spam classification using Streamlit. You can easily deploy this model to a cloud service or use it for real-time classification.
