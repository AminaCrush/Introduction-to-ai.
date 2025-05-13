import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

sample_message = """From ilug-admin@linux.ie Mon Jul 29 11:28:02 2002 Return-Path: <ilug-admin@linux.ie> Delivered-To: yyyy@localhost.netnoteinc.com Received: from localhost (localhost [127.0.0.1]) by phobos.labs.netnoteinc.com (Postfix)..."""

X = vectorizer.transform([sample_message])
prediction = model.predict(X)[0]

print("Message:")
print(sample_message[:300] + "...")
print(f"Prediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
