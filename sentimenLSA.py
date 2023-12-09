import requests
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# buat list kosong
twitter_data = []

# Mengambil Data dari Twitter Menggunakan Scraper API:
payload = {
    'api_key': '54d2215e72eccfbc84aa3a9c456e2559',
    'query': 'tiktok shop indonesia',
    'num': '100',
    'since': '2023-10-04',
    'until': '2023-10-11'}
response = requests.get('https://api.scraperapi.com/structured/twitter/search', params=payload)


# permintaan HTTP status code 200 menandakan bahwa permintaan telah berhasil
if response.status_code == 200:
    data = response.json()
else:
    print(f"Error in API request. Status code: {response.status_code}")

# ambil data asli tanpa manipulasi
all_tweets = data['organic_results']
for tweeting in all_tweets:
    twitter_data.append(tweeting)

# Preprocessing
preprocessed_tweets = []
for tweets in twitter_data:
    tweets = tweets.get('snippet', '')
    tweets = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweets)
    tweets  = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweets)
    tweets  = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweets)
    tweets  = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweets)
    tweets  = tweets.lower()
    tweets = re.sub(r"\W"," ",tweets)
    tweets = re.sub(r"\d"," ",tweets)
    tweets = re.sub(r"\s+[a-z]\s+"," ",tweets)
    tweets = re.sub(r"\s+[a-z]$"," ",tweets)
    tweets = re.sub(r"^[a-z]\s+"," ",tweets)
    tweets = re.sub(r"\s+"," ",tweets)
    preprocessed_tweets.append(tweets)

# Membuat Matriks TF-IDF untuk LSA
vectorizer_lsa = TfidfVectorizer(max_features=5000)
X_lsa = vectorizer_lsa.fit_transform(preprocessed_tweets)

# Melakukan Latent Semantic Analysis (LSA):
num_topics = 200
lsa = TruncatedSVD(n_components=num_topics,random_state=42)
X_lsa = lsa.fit_transform(X_lsa)

# Mendefinisikan label untuk setiap dokumen dalam matriks TF-IDF yang sudah di-LSA
labels = [1] * (len(X_lsa) // 3) + [0] * (len(X_lsa) // 3) + [2] * (len(X_lsa) // 3)
if len(X_lsa) % 3 == 1:
    labels.append(1)
elif len(X_lsa) % 3 == 2:
    labels.extend([0, 2])

# pemisah training 75% dan test 25% 
X_train, X_test, y_train, y_test = train_test_split(X_lsa, labels, test_size=0.2, random_state=42)
classifier_lsa = LogisticRegression(max_iter=200,random_state=42)  
classifier_lsa.fit(X_train, y_train)

# Evaluasi Model 
y_pred_lsa = classifier_lsa.predict(X_test)
accuracy_lsa = accuracy_score(y_test, y_pred_lsa)   
print(f"Accuracy (LSA): {accuracy_lsa}")
print(classification_report(y_test, y_pred_lsa,  zero_division=1))
objects = ['Negative', 'Neutral', 'Positive']
y_pos = np.arange(len(objects))
# Visualisasi Hasil:
plt.bar(y_pos, [np.sum(np.array(y_test) == 0), np.sum(np.array(y_test) == 1), np.sum(np.array(y_test) == 2)], alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number')
plt.title('Number of Negative, Neutral, and Positive Tweets (LSA)')

plt.show()

