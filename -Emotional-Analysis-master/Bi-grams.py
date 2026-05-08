import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import demoji
import nltk
import re
import math
import emoji
import demoji

df = pd.read_csv('new_dataset.csv')

def remove_emojis(text):
    return demoji.replace(text, '')

def demojize_text(text):
    return emoji.demojize(text)

df['sentence']=df['sentence'].apply(remove_emojis)

custom_stopwords = set()
with open('stopwords.txt', 'r', encoding='utf-8') as file:
    for line in file:
        custom_stopwords.add(line.strip())

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]
    return ' '.join(filtered_words)

df['sentence'] = df['sentence'].apply(remove_stopwords)

def normalize_text(text):
    text = text.lower()

    #Remove Special Characters.
    text = re.sub(r"[^\w\s]", " ", text)
    return text
    
df['sentence']=df['sentence'].apply(normalize_text)
text_data = df['sentence'].tolist()



def custom_tokenizer(text):
    words = text.split()
    result = []
    for i in range(len(words) - 1):
        if words[i] == "not":
            result.append("not " + words[i + 1])
    return result


def calculate_tf(document):
    tf = {}
    words = document.split()
    word_count = len(words)
    
    for word in words:
        tf[word] = tf.get(word, 0) + 1 / word_count
    
    return tf

# Calculate IDF
def calculate_idf(corpus):
    idf = {}
    doc_count = len(corpus)

    for document in corpus:
        words = set(document.split())
        for word in words:
            idf[word] = idf.get(word, 0) + 1
    
    for word, count in idf.items():
        idf[word] = math.log(doc_count / (count + 1))
    
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


ngram_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
X = ngram_vectorizer.fit_transform(text_data)


# Get feature names (custom bigrams)
bi_grams = ngram_vectorizer.get_feature_names_out()
bi_grams_df = pd.DataFrame(bi_grams, columns=['Bi-Grams'])
bi_grams_df.to_csv('bi_grams.csv', index=False)
# Display the generated n-grams
print("Bi-Grams with not :")
print(bi_grams)