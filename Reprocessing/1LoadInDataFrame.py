import pandas as pd

df = pd.read_csv("./data/dataset.txt",sep="/", names=["row"]).dropna()
print(df.head(7))