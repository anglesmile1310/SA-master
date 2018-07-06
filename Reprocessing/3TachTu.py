from nltk import ngrams
import pandas as pd
df = pd.read_csv("./data/ProcessRow.txt",sep="/", names=["row"]).dropna()
def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]

df["1gram"] = df.row.apply(lambda t: kieu_ngram(t, 1))
df["2gram"] = df.row.apply(lambda t: kieu_ngram(t, 2))

print(df.head(10))