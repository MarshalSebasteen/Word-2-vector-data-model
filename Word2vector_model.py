import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from gensim.models import Word2Vec

df = pd.read_csv(r"C:\Users\marsh\Downloads\amazon_reviews.csv")  

if 'reviewText' not in df.columns:
    raise ValueError("The input CSV must have a 'reviewText' column.")

df['reviewText'] = df['reviewText'].fillna('')  

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()                           
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)     
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_review'] = df['reviewText'].apply(clean_text)

tokenized_reviews = [word_tokenize(review) for review in df['cleaned_review'] if review.strip() != '']

print(tokenized_reviews[:5]) 

model = Word2Vec(
    sentences=tokenized_reviews,
    vector_size=100,   
    window=5,          
    min_count=2,      
    workers=2,         
    sg=1               
)

model.save("amazon_reviews_word2vec.model")
print("Word2Vec model saved successfully.")
