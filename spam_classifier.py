import string
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# LOAD DATASET SAFELY (NO PANDAS CSV PARSER)
# ------------------------------------------------
labels = []
messages = []

try:
    with open("sms_spam.csv", "r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, message = parts
                if label.lower() in ["spam", "ham"]:
                    labels.append(label.lower())
                    messages.append(message)
except FileNotFoundError:
    print("ERROR: sms_spam.csv file not found.")
    sys.exit()

print("Total Emails Loaded:", len(messages))

# STOP if dataset is empty
if len(messages) == 0:
    print("ERROR: Dataset is empty. Check file format.")
    sys.exit()

# Encode labels
y = [1 if label == "spam" else 0 for label in labels]

# ------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

X = [clean_text(msg) for msg in messages]

# ------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# FEATURE EXTRACTION (TF-IDF)
# ------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------------------------
# MODEL TRAINING (NAIVE BAYES)
# ------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------
print("\nSpam Email Classifier Ready")

while True:
    email = input("\nEnter email text (type 'exit'): ")

    if email.lower() == "exit":
        print("Program Ended")
        break

    email_clean = clean_text(email)
    email_vec = vectorizer.transform([email_clean])
    prediction = model.predict(email_vec)[0]

    if prediction == 1:
        print("Result: Spam Email")
    else:
        print("Result: Not Spam (Ham)")
