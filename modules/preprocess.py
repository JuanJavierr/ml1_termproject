import spacy
import re
import string
import nltk
from nltk.corpus import stopwords
import html 
from tqdm import tqdm

nltk.download('stopwords')
nlp = spacy.load("fr_core_news_sm")

def spacy_tokenizer(text):
    nlp = spacy.load("fr_core_news_sm")
    return [tok.text for tok in nlp.tokenizer(str(text))]

def text_edit(dataset, grp_num=False, rm_newline=False, rm_punctuation=False,
              rm_stop_words=False, lowercase=False, lemmatize=False, html_=False, convert_entities=False):

    stop_words = set(stopwords.words('french'))  
    pattern = re.compile(f"[{re.escape(string.punctuation)}]")  

    for attrs in tqdm(dataset.values()):
        text_ = attrs['text']

        if convert_entities:
            doc = nlp(text_)
            for ent in doc.ents:
                if ent.label_ in ['PERSON']:
                    text_ = text_.replace(ent.text, 'person')
                elif ent.label_ in ['GPE', 'LOC']:
                    text_ = text_.replace(ent.text, 'place')
                elif ent.label_ in ['DATE', 'TIME']:
                    text_ = text_.replace(ent.text, 'time')

        if html_:
            text_ = html.unescape(text_)

        if lowercase:
            text_ = text_.lower()

        if rm_stop_words:
            words = text_.split()
            text_ = ' '.join(word for word in words if word not in stop_words)

        if grp_num:
            text_ = re.sub(r'\d+', 'num', text_)

        if rm_newline:
            text_ = text_.replace('\n', '')

        if rm_punctuation:
            text_ = pattern.sub('', text_)
            text_ = re.sub(r' +', ' ', text_)

        if lemmatize:
            text_words = text_.split()
            text_ = ' '.join(tok.lemma_ for tok in nlp(' '.join(text_words)))

        attrs['text'] = text_

    return dataset