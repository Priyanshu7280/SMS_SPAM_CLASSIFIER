import pickle

# Load and inspect the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Loaded model:")
print(model)
print("Model type:", type(model))

# Load and inspect the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("\nLoaded vectorizer:")
print(vectorizer)
print("Vectorizer type:", type(vectorizer))

# Optional: Show feature names if vectorizer is a TfidfVectorizer
if hasattr(vectorizer, 'get_feature_names_out'):
    features = vectorizer.get_feature_names_out()
    print("\nSome feature names:", features[:10])
