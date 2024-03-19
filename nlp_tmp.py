# %%
import pandas as pd


df = pd.read_json("lapresse/output.json")

# %%
contents = df.loc[0]["text"]
# %%
import spacy

nlp = spacy.load("fr_dep_news_trf")

# %%
doc = nlp(contents)

# %%
for w in doc.sents:
    print(w)
    print("=" * 10)

# %%
