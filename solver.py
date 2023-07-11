import email
import re

import nltk
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

df = pd.read_csv('emails.csv')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("number of folders: ", df.shape)


def get_field(field, messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))
    return column


def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column


df['body'] = body(df['message'])
df['body'] = df['body'].str.lower()
df['body'] = df['body'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
df['body'] = df['body'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
df['body'] = df['body'].apply(lambda x: [word for word in x if word not in stop_words])
lemmatizer = WordNetLemmatizer()
df['body'] = df['body'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
ndf = df['body']
ndf.to_csv('preprocessed_email_dataset_1.csv', index=False)

print("number of folders: ", df.shape[0])

# # Lowercase the text
# df['text'] = df['text'].str.lower()
#
# # Tokenize the text
# df['text'] = df['text'].apply(word_tokenize)
#
# # Remove stopwords
# stop_words = set(stopwords.words('english'))
# df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])
#
# # Lemmatize the text
# lemmatizer = WordNetLemmatizer()
# df['text'] = df['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
#
# df.to_csv('preprocessed_email_dataset.csv', index=False)
