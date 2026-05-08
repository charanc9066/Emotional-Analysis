import pandas as pd
import numpy as np
import re
import math
import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import demoji
from sklearn.neural_network import MLPClassifier

#1: 6052values
# 0: 5053values
# 3: 2649values
# 4: 1933values
# 2: 1299values
# 5: 568values
#,nrows=10
emotion = pd.read_csv("new_dataset.csv")
print(emotion.shape)
# print(emotion.columns)
emotion = emotion.drop_duplicates(subset='sentence', keep="first")
print(emotion.shape)

# emotion['emotion'].replace({'neutral':0,'worry':1,'happiness':2,'sadness':3,'love':4,'surprise':5,'fun':6,'relief':7,'hate':8,'empty':9,'enthusiasm':10,'boredom':11,'anger':12},inplace=True)
emotion_counts = emotion['emotion'].value_counts()

# Print the counts in a neat fashion
# for emotion, count in emotion_counts.items():
#     print(f"{emotion}: {count} values")

def demojize_text(text):
    return emoji.demojize(text)

# emotion['sentence']=emotion['sentence'].apply(demojize_text)
def remove_emojis(text):
    return demoji.replace(text, '')


emotion['sentence']=emotion['sentence'].apply(remove_emojis)

# print(emotion["meaning"])

#Cleaning html,links and other noise
def clean_html(text):
    
  # Remove links
  text = re.sub(r"https?:\/\/\S+\b", "", text)
  # Remove irrelevant characters.
  text = re.sub(r"[^\x00-\x7F]", "", text)
  # Remove HTML tags.
  text = re.sub(r"<[^>]*>", "", text)
  #Remove Character mentions.
  text = re.sub(r"@\w+", "", text)
  #Remove Alpha Numeric words
  text = re.sub(r'\w*\d\w*', ' ', text)
  #Remove numbers
  text = re.sub(r'\d+', '', text)
  # Remove whitespace.
  text = text.strip()

  return text
emotion['sentence']=emotion['sentence'].apply(clean_html)
# emotion['meaning']=emotion['meaning'].apply(clean_html)
# print(emotion)

expand_contractions = {
    "can't": "cannot",
    "won't": "will not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "it's": "it is",
    "i'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "i'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "aren't": "are not",
    "weren't": "were not",
    "can't've": "cannot have",
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    "ain't": "am not",
    "isn't": "is not",
    "aren't": "are not",
    "hasn't": "has not",
    "haven't": "have not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
}

#Normalization
def normalize_text(text):
    text = text.lower()

    for word in expand_contractions:
       text = re.sub(r'\b{}\b'.format(word), expand_contractions[word], text)

    #Remove Special Characters.
    text = re.sub(r"[^\w\s]", " ", text)
    return text
    
emotion['sentence']=emotion['sentence'].apply(normalize_text)
# emotion['meaning']=emotion['meaning'].apply(normalize_text)
# print(emotion)

#Stop Word Removal
custom_stopwords = set()
with open('stopwords.txt', 'r', encoding='utf-8') as file:
    for line in file:
        custom_stopwords.add(line.strip())

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]
    return ' '.join(filtered_words)

emotion['sentence'] = emotion['sentence'].apply(remove_stopwords)
# emotion['meaning']=emotion['meaning'].apply(remove_stopwords)

# print(emotion)

lemmatization_rules = {}
with open("lemmatization.txt", "r") as rules_file:
    for line in rules_file:
        from_word, to_word = line.strip().split("\t")
        lemmatization_rules[from_word] = to_word

def lemmatize_text(sentence):
    words = sentence.split()
    lemmatized_words = [lemmatization_rules.get(word, word) for word in words]
    return ' '.join(lemmatized_words)


emotion['sentence']=emotion['sentence'].apply(lemmatize_text)
# emotion['meaning']=emotion['meaning'].apply(lemmatize_text)
# print(emotion)

def find_nearest_dictionary_word(word):
 
  dictionary_words = words.words()

  # Calculate the edit distance between the non-dictionary word and each dictionary word.
  edit_distances = []
  for dictionary_word in dictionary_words:
    edit_distances.append(edit_distance(word, dictionary_word))

  # Find the dictionary word with the smallest edit distance.
  nearest_dictionary_word = dictionary_words[edit_distances.index(min(edit_distances))]

  return nearest_dictionary_word

def replace_non_dictionary_words(text):

  # Split the text into words.
  words_list = text.split()

  # Replace all non-dictionary words.
  for i in range(len(words_list)):
    word = words_list[i]
    if word not in words.words():
      nearest_dictionary_word = find_nearest_dictionary_word(word)
      words_list[i] = nearest_dictionary_word

  # Join the words back into a string.
  processed_text = " ".join(words_list)

  return processed_text

# emotion['sentence']=emotion['sentence'].apply(replace_non_dictionary_words)
# emotion['meaning']=emotion['meaning'].apply(replace_non_dictionary_words)
# print(emotion)

# sentences = emotion['sentence']

# # Initialize a TF-IDF vectorizer
# tfidf_vectorizer = TfidfVectorizer()

# # Fit the vectorizer on the sentences and transform them into TF-IDF vectors
# tfidf_vectors = tfidf_vectorizer.fit_transform(sentences)

# # Get the TF-IDF features as a dense array
# tfidf_features = tfidf_vectors.toarray()

# # Print the TF-IDF features for each word in each sentence
# terms = tfidf_vectorizer.get_feature_names_out()

# for i, sentence in enumerate(sentences):
#     print("Sentence:", sentence)
#     for j, term in enumerate(terms):
#         tfidf_weight = tfidf_features[i][j]
#         if tfidf_weight > 0:
#             print(f"{term}: {tfidf_weight:.4f}")
#     print()
tfidf_vectorizer = TfidfVectorizer()

X = emotion['sentence']
y = emotion['emotion']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, you can print the sizes of the resulting sets
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))



# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Optionally, you can convert the result to a dense array for better readability
# X_train_tfidf_array = X_train_tfidf.toarray()

# Print the vocabulary (feature names) and the TF-IDF matrix
# print("Vocabulary (Feature Names):", tfidf_vectorizer.get_feature_names_out())
# print("TF-IDF Matrix:")
# print(X_train_tfidf_array)

vocabulary = tfidf_vectorizer.get_feature_names_out()
tfidf_values = X_train_tfidf.toarray()
print(len(vocabulary))
# # Create a DataFrame with feature names and TF-IDF values
tfidf_df = pd.DataFrame(tfidf_values, columns=vocabulary)
non_zero_tfidf_df = tfidf_df.loc[(tfidf_df != 0).any(axis=1)]
# Print the feature names and their corresponding TF-IDF values
# print("Vocabulary (Feature Names):", vocabulary)
print("TF-IDF Matrix:")
print(non_zero_tfidf_df)



def calculate_tf(document):
    tf = {}
    words = document.split()
    word_count = len(words)
    
    for word in words:
        tf[word] = tf.get(word, 0) / word_count
    
    return tf

# Calculate IDF
def calculate_idf(corpus):
    idf = {}
    doc_count = len(corpus)

    for document in corpus:
        words = set(document.split())
        for word in words:
            idf[word] = idf.get(word, 0)
    
    for word, count in idf.items():
        idf[word] = math.log(doc_count / (count))
    
    return idf

# Calculate TF-IDF
def calculate_tfidf(corpus):
    tfidf = []
    tf_matrix = []

    for document in corpus:
        tf = calculate_tf(document)
        tf_matrix.append(tf)

    idf = calculate_idf(corpus)

    for tf in tf_matrix:
        tfidf_doc = {}
        for word, value in tf.items():
            tfidf_doc[word] = value * idf[word]
        tfidf.append(tfidf_doc)

    return tfidf

class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha 
        self.class_probs = None
        self.feature_probs = None
        self.classes = None

    def fit(self, X_train, y_train):
        # Get unique classes and calculate class probabilities
        self.classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        self.class_probs = class_counts / total_samples

        # Calculate feature probabilities for each class
        self.feature_probs = {}
        for cls in self.classes:
            cls_indices = np.where(y_train == cls)
            cls_features = X_train[cls_indices]
            feature_counts = np.sum(cls_features, axis=0)
            total_features = np.sum(feature_counts)
            probs = (feature_counts + self.alpha) / (total_features + len(feature_counts)*self.alpha)
            self.feature_probs[cls] = probs

    def predict(self, X_test):
      predictions = []
      for sample in X_test:
          posteriors = []
          for i, cls in enumerate(self.classes):
              class_prob = np.log(self.class_probs[i])
              feature_prob = np.sum(np.log(self.feature_probs[cls]) * sample)
              posterior = class_prob + feature_prob
              posteriors.append(posterior)
          predictions.append(self.classes[np.argmax(posteriors)])
      return predictions

# Train a classifier (e.g., Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Train the custom classifier
custom_classifier = CustomMultinomialNB()
custom_classifier.fit(X_train_tfidf.toarray(), y_train)

# Make predictions on the test set
y_custom_pred = custom_classifier.predict(X_test_tfidf.toarray())
# print(y_custom_pred)

# Evaluate the custom model
accuracy_custom = accuracy_score(y_test, y_custom_pred)
report_custom = classification_report(y_test, y_custom_pred)
confusion_custom = confusion_matrix(y_test,y_custom_pred)

print("Accuracy : " ,accuracy_custom)
print("Classification Report:")
print(report_custom)
emotion_mapping = {
    '0': "sad",
    '1': "happy",
    '2': "love",
    '3': "anger",
    '4': "fear",
    '5': "surprise"
}

#"Today has been a challenging day, filled with a heavy heart and a sense of melancholy, as I've been grappling with some personal struggles; now, I'm reaching out to you, my dear friend, seeking solace and understanding, hoping for your comforting words and support to guide me through this difficult time."

query = input("Enter your query :")
new_query_tfidf = tfidf_vectorizer.transform([query])

# Predict the emotion for the new query
predicted_emotion = classifier.predict(new_query_tfidf)

predicted_emotion_text = emotion_mapping[predicted_emotion[0]]
# Print the predicted emotion
print("Predicted Emotion:",predicted_emotion_text)

