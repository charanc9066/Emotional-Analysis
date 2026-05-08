import pandas as pd
import numpy as np
import re
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
from nltk.tokenize import word_tokenize
from nltk import pos_tag

#,nrows=10
emotion = pd.read_csv("new_dataset.csv")
print(emotion.shape)
# print(emotion.columns)
emotion = emotion.drop_duplicates(subset='sentence', keep="first")
print(emotion.shape)

# emotion['emotion'].replace({'neutral':0,'worry':1,'happiness':2,'sadness':3,'love':4,'surprise':5,'fun':6,'relief':7,'hate':8,'empty':9,'enthusiasm':10,'boredom':11,'anger':12},inplace=True)

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


X = emotion['sentence']
y = emotion['emotion']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test.to_csv('testing_data.csv', index=False)

# Optionally, you can print the sizes of the resulting sets
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

#Oversampling of sad emotion

file1 = "new_dataset.csv"
file2 = "sad_dataset.csv"
output_file = "new_data.csv"

def concatenate_csv_files(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    # Save the result to a new CSV file
    concatenated_df.to_csv(output_file, index=False)

def detect_emotion(query, emotions):
    # Initialize emotion scores
    emotion_scores = {emotion: 0 for emotion in emotions}

    # Tokenize the query (you can replace this with your specific preprocessing steps)
    words = query.lower().split()

    # Initialize a flag for negation
    negate = False

    # Check each word in the query
    for word in words:
        # Check for negation
        if word == "not":
            negate = True
            continue

        # Check if the word is present in any emotion dictionary
        for emotion, word_dict in emotions.items():
            if negate and word in word_dict:
                # Subtract the weighted score of the word if it follows a "not"
                emotion_scores[emotion] -= word_dict[word]
            elif word in word_dict:
                # Add the weighted score of the word to the corresponding emotion
                emotion_scores[emotion] += word_dict[word]

        # Reset the negation flag
        negate = False

    # Determine the emotion with the highest score
    detected_emotion = max(emotion_scores, key=emotion_scores.get)

    return detected_emotion


def extract_adverbs_adjectives(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    adverbs = [word for word, pos in tagged_words if pos.startswith('RB')]  # Extracting adverbs
    adjectives = [word for word, pos in tagged_words if pos.startswith('JJ')]  # Extracting adjectives
    return adverbs, adjectives

# Apply the extraction function to the 'sentence' column
emotion[['adverbs', 'adjectives']] = emotion['sentence'].apply(lambda x: pd.Series(extract_adverbs_adjectives(x)))
# print(emotion[['adverbs', 'adjectives']])

sad_dict = {
    'sad': 0.9, 'unhappy': 0.7, 'melancholy': 0.9, 'grief': 0.8, 'despair': 0.7, 'tearful': 0.8, 'heartache': 0.7,
    'mournful': 0.9, 'dejected': 0.8, 'sorrow': 0.7, 'downcast': 0.8, 'woeful': 0.9, 'disheartened': 0.7, 'hopeless': 0.8,
    'depressed': 0.9, 'melancholy': 0.8, 'dismayed': 0.7, 'pathetic': 0.8, 'hostile': 0.9, 'dissatisfied': 0.8,
    'resigned': 0.7, 'unwelcome': 0.8, 'uncomfortable': 0.9, 'humiliated': 0.8, 'irritable': 0.7, 'faking': 0.8,
    'unhappy': 0.9, 'heartbroken': 0.8, 'threatened': 0.7, 'rejected': 0.8, 'isolated': 0.9, 'helpless': 0.8,
    'disappointed': 0.7, 'guiltiest': 0.8, 'terrified': 0.9, 'saddest': 0.8, 'numbs': 0.7, 'anguished': 0.8,
    'miserable': 0.9, 'wretched': 0.8, 'gloomy': 0.7, 'forelorn': 0.8
}
happy_dict = {
   'happy' :0.6 ,  'joy': 0.9, 'elation': 0.8, 'cheerful': 1.0, 'uplifting': 0.9, 'contentment': 0.7, 'bliss': 0.8, 'satisfaction': 0.7,
    'glad': 0.7, 'pleasure': 0.8, 'smile': 0.9, 'laughter': 1.0, 'vibrant': 0.8, 'radiant': 0.7, 'thrilled': 0.9,
    'gratitude': 1.0, 'festive': 0.8, 'exhilaration': 0.9, 'euphoria': 1.0, 'merry': 0.7, 'jovial': 0.8,
    'celebration': 0.9, 'positive': 1.0, 'enthusiastic': 0.9, 'blessed': 0.8, 'harmony': 0.7, 'delightful': 0.8,
    'ecstatic': 1.0, 'amusement': 0.9, 'optimistic': 0.9, 'serenity': 0.7, 'playful': 0.8, 'vibe': 1.0, 'bashings': 0.8,
    'jubilant': 0.9, 'generous': 0.7, 'relieved': 0.8, 'excited': 1.0, 'fabulous': 0.9, 'energetic': 0.8, 'joyful': 0.9,
    'grateful': 0.7, 'surprised': 0.8, 'delighted': 0.9, 'honored': 0.8, 'successful': 0.9, 'supportive': 1.0,
    'funniest': 0.8, 'amused': 0.7, 'appreciative': 0.8, 'ecstatic': 0.9, 'blessed': 1.0, 'privileged': 0.8,
    'loved': 0.9, 'compassionate': 0.7, 'upbeat': 0.8, 'inspired': 1.0, 'energetic': 0.9, 'satisfied': 0.8,
    'relaxed': 0.7, 'peaceful': 0.8, 'optimistic': 0.9, 'thankful': 1.0, 'celebrating': 0.8, 'good':0.6
}

fear_dict = {
    'fear': 0.9, 'anxiety': 0.8, 'dread': 0.7, 'worry': 0.8, 'apprehension': 0.9, 'terror': 0.8, 'panic': 0.7,
    'nightmares': 0.8, 'worst': 0.9, 'unsafe': 0.8, 'hostile': 0.7, 'threatened': 0.8, 'frightened': 0.9,
    'intimidated': 0.8, 'frightened': 0.7, 'fearful': 0.8, 'terrified': 0.9, 'helpless': 0.8, 'doubts': 0.7,
    'stressing': 0.8, 'griefs': 0.9, 'resolved': 0.8, 'paranoid': 0.7, 'worried': 0.8, 'awkward': 0.9, 'lonely': 0.8,
    'restless': 0.7, 'fearful': 0.8, 'anxious': 0.9, 'panicky': 0.8, 'worries': 0.7, 'insecurity': 0.8,
    'nervousness': 0.9, 'shivers': 0.8, 'apprehensive': 0.7, 'foreboding': 0.8, 'disquiet': 0.9, 'uneasiness': 0.8,'afraid':0.8
}

surprise_dict = {
    'surprise': 0.8, 'astonishment': 0.9, 'amazement': 0.7, 'shock': 0.8, 'startled': 0.9, 'unexpected': 0.8,
    'discovery': 0.7, 'incredible': 0.8, 'awe': 0.9, 'wonder': 0.8, 'unexpectedly': 0.7, 'astonished': 0.8,
    'amazed': 0.9, 'stunned': 0.8, 'astounded': 0.7, 'shocked': 0.8, 'honestly': 0.9, 'startled': 0.8, 'amazed': 0.7,
    'astonished': 0.8, 'unexpected': 0.9, 'discovering': 0.8, 'inexplicably': 0.7, 'burst': 0.8, 'awe': 0.9,
    'wonder': 0.8, 'unexpectedly': 0.7
}

love_dict = {
    'love': 1.0, 'affection': 0.9, 'devotion': 0.8, 'passion': 0.9, 'romance': 0.8, 'admiration': 0.9, 'positive': 0.8,
    'vibe': 1.0, 'bashings': 0.7, 'closed': 0.8, 'positives': 0.9, 'sharing': 0.8, 'blog': 0.7, 'braving': 0.8,
    'walks': 0.9, 'perfuming': 0.8, 'scented': 0.7, 'encounters': 0.8, 'successful': 0.9, 'matters': 0.8,
    'competitors': 0.7, 'considerate': 0.8, 'caring': 0.9, 'owners': 0.8, 'whiney': 0.7, 'conversations': 0.8,
    'admitting': 0.9, 'disheartened': 0.8, 'disillusioned': 0.7, 'publishings': 0.8, 'communities': 0.9,
    'doubts': 0.8, 'stressing': 0.7, 'griefs': 0.8, 'resolved': 0.9, 'gorgeous': 0.8, 'envying': 0.7, 'settled': 0.8,
    'lives': 0.9, 'living': 0.8, 'prettiest': 0.7, 'sympathetic': 0.8, 'defeated': 0.9, 'hears': 0.8,
    'nostalgia': 0.7, 'floods': 0.8, 'jubilant': 0.9, 'generous': 0.8, 'shocked': 0.7, 'honestly': 0.8, 'maddest': 0.9,
    'graceful': 0.8, 'loving': 0.9, 'smiling': 0.8, 'nurturing': 0.7, 'heals': 0.8, 'happiness': 0.9, 'contagious': 0.8,
    'sparking': 0.7, 'spreads': 0.8, 'glows': 0.9, 'mellows': 0.8, 'positive': 0.7, 'vocals': 0.8, 'convinced': 0.9,
    'walks': 0.8, 'opinions': 0.7, 'matters': 0.8, 'playful': 0.9, 'danced': 0.8, 'reliefs': 0.7, 'aches': 0.8, 'tad': 0.9,
    'overwhelmed': 0.8, 'currents': 0.7, 'livings': 0.8, 'situations': 0.9, 'peaceful': 0.8, 'dulls': 0.7, 'lacks': 0.8,
    'excitements': 0.9, 'flowers': 0.8, 'fragrances': 0.7, 'crushes': 0.8, 'generous': 0.9, 'packed': 0.8,
    'pressured': 0.7, 'perfects': 0.8, 'fitting': 0.9, 'audiences': 0.8, 'gorgeous': 0.7, 'creativity': 0.8, 'cools': 0.9,
    'incredibly': 0.8, 'agitated': 0.7, 'exposing': 0.8, 'vulnerablest': 0.9, 'dresses': 0.8, 'cools': 0.7,
    'naughtiest': 0.8, 'prayers': 0.9, 'reluctant': 0.8, 'woken': 0.7, 'grumpiest': 0.8, 'tired': 0.9,
    'unhappiest': 0.8, 'plain': 0.7, 'sickest': 0.8, 'confident': 0.9, 'india': 0.8, 'breathings': 0.7, 'keeping': 0.8,
    'awoken': 0.9, 'beatings': 0.8, 'paranoids': 0.7, 'tonight': 0.8, 'checks': 0.7, 'stats': 0.8, 'satisfied': 0.9,
    'unresponsive': 0.8, 'unloved': 0.7, 'hurting': 0.8, 'bits': 0.9
}
anger_dict = {
    'anger': 0.9, 'rage': 0.8, 'irritation': 0.7, 'frustration': 0.8, 'fury': 0.9, 'resentment': 0.8, 'hostility': 0.7,
    'outrage': 0.8, 'indignation': 0.9, 'annoyance': 0.8, 'enraged': 0.7, 'exasperation': 0.8, 'agitation': 0.9,
    'displeasure': 0.8, 'provoked': 0.7, 'infuriated': 0.8, 'aggravation': 0.9, 'impatience': 0.8, 'fuming': 0.7,
    'incensed': 0.8, 'irascible': 0.9, 'disgust': 0.8, 'irate': 0.7, 'upset': 0.8, 'mad': 0.9, 'offended': 0.8,
    'livid': 0.7, 'exasperated': 0.8, 'aggressive': 0.9, 'tense': 0.8, 'bitterness': 0.7, 'wrath': 0.8,
    'provocation': 0.9, 'irate': 0.8, 'infuriation': 0.7
}
# # Example query
# query = "Iam unhappy!"

#The blatant disregard for my suggestions, coupled with the insolent attitude of my colleagues, has triggered an overwhelming sense of fury and indignation within me. Their lack of consideration and the audacity to dismiss my input have left me fuming with a mix of resentment and exasperation. This level of provocation and insensitivity is simply intolerable, and I find myself on the verge of an outburst, teetering between a boiling rage and an urge to express my aggressive displeasure.
# result = detect_emotion(query, emotions)
# print("Detected Emotion:", result)

emotions = {'happy': happy_dict, 'sad': sad_dict, 'fear': fear_dict, 'surprise': surprise_dict, 'love': love_dict,'anger':anger_dict}

emotion_labels = {
    'sad':0,
    'happy':1,
    'love':2,
    'anger':3,
    'fear':4,
    'surprise':5
}
def map_emotions(label):
    return emotion_labels[label]

# Map numerical labels to emotions for 'actual_emotion' and 'detected_emotion' columns

emotion['detected_emotion'] = emotion['sentence'].apply(lambda x: detect_emotion(x, emotions))
# print(len (emotion["detected_emotion"]=='happy'))
emotion['detected_emotion'] = emotion['detected_emotion'].apply(map_emotions)

#print(emotion["detected_emotion"])
# Calculate accuracy
correct_predictions = (emotion['emotion'] == emotion['detected_emotion']).sum()
total_predictions = len(emotion)
accuracy = correct_predictions / total_predictions

#The blatant disregard for my suggestions, coupled with the insolent attitude of my colleagues, has triggered an overwhelming sense of fury and indignation within me. Their lack of consideration and the audacity to dismiss my input have left me fuming with a mix of resentment and exasperation. This level of provocation and insensitivity is simply intolerable, and I find myself on the verge of an outburst, teetering between a boiling rage and an urge to express my aggressive displeasure.

query = input("Enter your query : ")
result = detect_emotion(query, emotions)
print("Detected Emotion:", result)
