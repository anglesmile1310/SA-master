# Training gensim model
from gensim.models import Word2Vec
from nltk import ngrams
import logging
import pandas as pd
from pyvi import ViTokenizer
df = pd.read_csv("./data/ProcessRow.txt",sep="/", names=["row"]).dropna()
#.dropna(): remove missing value
def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]
print(len(df))
with open("./data/Tokenize.txt",'w+',encoding="UTF-8") as file:
    for i in range(len(df)):
        file.write(ViTokenizer.tokenize((str(df.row[i])))+"\n")
df1=pd.read_csv("./data/Tokenize.txt",sep='/',names=["row"]).dropna()
df1["PyVi"] = df1.row.apply(lambda t: kieu_ngram(t, 1))
#.apply(): các chức năng dọc theo các trục của dataframe
#lambda :hàm vô danh là hàm được định nghĩa mà không có tên,tương tự như def.
print(df1["PyVi"])
train_data=df1["PyVi"].tolist()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = Word2Vec(train_data, size=100, window=10, min_count=3, workers=4, sg=1)
print(model.wv.most_similar("tâm"))
model.train(train_data, total_examples=len(train_data), epochs=10)
print(model.wv.similar_by_word("tâm"))
#print(model.wv.most_similar("tài"))