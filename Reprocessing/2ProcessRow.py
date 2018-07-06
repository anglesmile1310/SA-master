import pandas as pd
import re
df = pd.read_csv("./data/dataset.txt",sep="/", names=["row"]).dropna()

def transform_row(row):
    # Xóa số dòng ở đầu câu
    row = re.sub(r"^[0-9\.]+", "", row)

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$", "", row)

    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ")

    row = row.strip()
    return row



df["row"] = df.row.apply(transform_row)
print(df["row"][0])
print(len(df["row"]))
with open("./data/ProcessRow.txt",'w+',encoding="UTF-8") as file:
    for i in range(len(df["row"])):
        file.write(str(df["row"][i])+"\n")
    file.close()
