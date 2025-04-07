import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
# You can download the dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv(r"C:\Users\Rathinavel\OneDrive\文档\spam.csv", encoding='latin-1')
df.columns = ['label', 'message']

# 2. Encode labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4. Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Predict & Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
if accuracy_score(y_test,y_pred)>=0.5:
    print( 'spam')
else:
    print('ham')
