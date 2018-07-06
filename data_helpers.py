import numpy as np
import re
from pyvi import ViTokenizer
import pandas as pd
import string
import itertools
from collections import Counter

# data dict_abbreviation
filename = './data1/dict/dict_abbreviation.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_abbreviation = data['abbreviation']
list_converts = data['convert']
#data stopword
filename1='./data1/dict/stopwords.csv'
data1=pd.read_csv(filename1,sep="\t",encoding='utf-8')
list_stopwords=data1['stopwords']
#Tiền xử lý dữ liệu
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#hàm này đọc dataset raw và tách chuỗi câu
def readdata(path):
    data = []
    with open(path, 'r',encoding="UTF-8") as f:
        rawdata = f.read().splitlines() #Hàm này sẽ tách chuỗi bởi các ký tự \n.(tách chuỗi theo dòng)
        for i in rawdata:
            if len(i.strip())==0:
                rawdata.remove(i)
    """
    for onecomment in rawdata:
        data.append(onecomment.split(':', 1)) #phân tách chuỗi trong dòng bởi ':', 1 là tách chuỗi 1 lần
    X = [data[i][1] for i in range(len(data))]  #chứa câu
    Y = [data[i][0] for i in range(len(data))]  #chứa chuỗi nhãn 0:là neutral,1:là negative,2:là positive
    """
    return [rawdata[i] for i in range(len(rawdata))]


def clean_data(comment):
    # loai link lien ket
    comment = re.sub(r'\shttps?:\/\/[^\s]*\s+|^https?:\/\/[^\s]*\s+|https?:\/\/[^\s]*$', ' link_spam ', comment)
    #chuyển hết link trong comment thành "link_spam"
    return comment

def convert_Abbreviation(comment):
    comment = re.sub('\s+', " ", comment)
    for i in range(len(list_converts)):
        abbreviation = '(\s' + list_abbreviation[i] + '\s)|(^' + list_abbreviation[i] + '\s)|(\s' \
                       + list_abbreviation[i] + '$)'
        convert = ' ' + str(list_converts[i]) + ' '
        comment = re.sub(abbreviation, convert, comment)

    return comment

def remove_Stopword(comment):
    re_comment = []
    words = comment.split()
    for word in words:
        if (not word.isnumeric()) and len(word) > 1 and word not in list_stopwords:
            re_comment.append(word)
    comment = ' '.join(re_comment)
    return comment


def tokenize(comment):
    text_token = ViTokenizer.tokenize(comment)
    return text_token

def normalize_Text(comment):
    comment = comment.encode().decode()
    comment = comment.lower()

    # thay gia tien bang text
    moneytag = [u'k', u'đ', u'ngàn', u'nghìn', u'usd', u'tr', u'củ', u'triệu', u'yên']
    for money in moneytag:
        comment = re.sub('(^\d*([,.]?\d+)+\s*' + money + ')|(' + '\s\d*([,.]?\d+)+\s*' + money + ')', ' monney ',
                         comment)
    comment = re.sub('(^\d+\s*\$)|(\s\d+\s*\$)', ' monney ', comment)
    comment = re.sub('(^\$\d+\s*)|(\s\$\d+\s*\$)', ' monney ', comment)

    # loai dau cau: nhuoc diem bi vo cau truc: vd; km/h. V-NAND
    listpunctuation = string.punctuation
    for i in listpunctuation:
        comment = comment.replace(i, ' ')

    # thay thong so bang specifications
    comment = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', comment)
    comment = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', comment)

    # thay thong so bang text lan 2
    comment = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', comment)
    comment = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', comment)

    # xu ly lay am tiet
    comment = re.sub(r'(\D)\1+', r'\1', comment)

    # #them dau cho nhung cau khong dau
    # words = comment.split()
    # for word in words:
    #     try:
    #         comment.encode('utf-8')
    #         word = accent.accent_comment(word)
    #         print word
    #     except UnicodeError:
    #         word = word.decode('utf-8')
    #     words_normal.append(word)

    # # them dau tach thanh tung cau
    # sents = re.split("([.?!])?[\n]+|[.?!] ", comment)
    # listpunctuation = string.punctuation
    # sents_normal = []
    # # them dau theo cau
    # for sent in sents:
    #     if sent != None:
    #         for i in listpunctuation:
    #             sent = sent.replace(i, " ")
    #         sent = accent.accent_comment(sent.decode('utf-8'))
    #         sents_normal.append(sent.lower())
    return comment

def predata(path):
    X = readdata(path)
    X_re = []
    i = 0
    for comment in X:
        comment = remove_Stopword(tokenize(convert_Abbreviation(normalize_Text(clean_data(comment)))))
        X_re.append(comment)

    return X_re #X_re chứa comment đã tiền xử lý

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    """
    # positive_examples=list các (câu ~ một dòng)
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples] #xóa backspace trong câu

    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]  #xóa backspace trong câu

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text] #Tiền xử lý các câu(dòng)
    """
    positive_examples =predata(positive_data_file)
    negative_examples=predata(negative_data_file)
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    # num_batches_per_epoch
    for epoch in range(num_epochs):
        # Shufle the data at each epoch,trộn data ở mỗi epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            #np.random.permutation():đảo ngẫu nhiên các element trong mảng
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            # generator là một hàm trả kết quả về là một chuỗi kết quả thay vì một giá trị duy nhất.
            #Mỗi lần lệnh yield được chạy, nó sẽ sinh ra một giá trị mới. (Vì thế nó mới được gọi là generator)

