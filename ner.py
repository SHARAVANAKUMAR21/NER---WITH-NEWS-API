import requests
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from typing import List, Tuple

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt data...")
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading averaged_perceptron_tagger data...")
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    print("Downloading maxent_ne_chunker data...")
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading words data...")
    nltk.download('words')

# Fetch news article from News API
def fetch_news_article(api_key: str, query: str = 'technology', language: str = 'en') -> str:
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'apiKey': api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        articles = data.get('articles', [])
        if articles:
            return articles[0].get('content', '')
        else:
            print("No articles found.")
            return ''
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ''

# Extract entities using SpaCy
def extract_entities_spacy(text: str) -> List[Tuple[str, str]]:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Extract entities using NLTK
def extract_entities_nltk(text: str) -> List[Tuple[str, str]]:
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunks = ne_chunk(tagged)
        
        entities = []
        for chunk in chunks:
            if isinstance(chunk, nltk.Tree):
                entity = ' '.join([leaf[0] for leaf in chunk.leaves()])
                label = chunk.label()
                entities.append((entity, label))
        return entities
    except Exception as e:
        print(f"An error occurred during NLTK processing: {e}")
        return []

# Compare results from SpaCy and NLTK
def compare_entities(text: str):
    spacy_entities = extract_entities_spacy(text)
    nltk_entities = extract_entities_nltk(text)
    
    print("SpaCy Entities:")
    for entity in spacy_entities:
        print(entity)
    
    print("\nNLTK Entities:")
    for entity in nltk_entities:
        print(entity)

# Example usage
if __name__ == "__main__":
    # Use your actual News API key
    api_key = '32f27771946b4251af72ff78a3a33c2c'
    article_text = fetch_news_article(api_key)
    
    if article_text:
        print("Article Content:\n", article_text[:1000])  # Print the first 1000 characters of the article
        compare_entities(article_text)
    else:
        print("Failed to fetch the article.")
