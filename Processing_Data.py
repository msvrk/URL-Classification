import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')


def url2text(url):
    page = requests.get(url)  # to extract page from website
    html_code = page.content  # to extract html code from page
    soup = BeautifulSoup(html_code, 'html.parser')  # Parse html code
    texts = soup.findAll(text=True)  # find all text
    text_from_html = ' '.join(texts)  # join all text
    return text_from_html


def initial_preprocessing_df():
    data.Text = data.Text.map(lambda p: re.sub(r'\W', ' ', p))  # Removing punctuations
    data.Text = data.Text.map(lambda p: p.lower())  # Converting to lowercase
    data.Text = data.Text.map(lambda p: re.sub(r'\s+[a-zA-Z0-9]\s+', ' ', p))  # Remove all single characters
    data.Text = data.Text.map(
        lambda p: re.sub(r'\s+', ' ', p, flags=re.I))  # Substituting multiple spaces with single space
    data.Text = data.Text.map(
        lambda df: RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?").tokenize(df.lower()))  # Tokenizing


def preprocessingSingleText(single_text):
    single_text = re.sub(r'\W', ' ', single_text)
    single_text = single_text.lower()
    single_text = re.sub(r'\s+[a-zA-Z0-9]\s+', ' ', single_text)
    single_text = re.sub(r'\s+', ' ', single_text, flags=re.I)
    single_text = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?").tokenize(single_text.lower())
    for token in single_text:
        if token in stopwords.words('english'):
            single_text.remove(token)
    stemmer = PorterStemmer()
    stemmed_text = []
    for token in single_text:
        stemmed_text.append(stemmer.stem(token))
    single_text = stemmed_text
    return single_text


def remove_stopwords(row):
    print(len(row.ExpText))
    for token in row.ExpText:
        if token in stopwords.words('english'):
            row.ExpText.remove(token)
    print(len(row.ExpText))
    return row


def stemming(index):
    stemmer = PorterStemmer()
    stemmed_text = []
    for t in data.Stemmed_text[index]:
        stemmed_text.append(stemmer.stem(t))
    data.Stemmed_text[index] = stemmed_text


def lemmatization(index):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    for t in data.Lemmatized_text[index]:
        lemmatized_text.append(lemmatizer.lemmatize(t))
    data.Lemmatized_text[index] = lemmatized_text

#
# # Reading Data
#
# data = pd.read_csv("URL_Dataset.csv")  # Dataset created manually from the web by listing different types of URLs
#
# print("Data successfully read")
#
# # Generating Text
# text = []
# for i in data.iloc[:, 0]:
#     text.append(url2text(i))
#
# data["PageContent"] = text
# data["Text"] = text
#
# print("Text successfully generated")
#
# # Text Preprocessing
#
# initial_preprocessing_df()
#
# data.apply(remove_stopwords, axis='columns')  # Removing stopwords
#
# data["Stemmed_text"] = data.Text
# for index in range(0, data.shape[0]):
#     stemming(index)
#
# data["Lemmatized_text"] = data.Text
# for index in range(0, data.shape[0]):
#     lemmatization(index)
#
# print("Text successfully preprocessed")
#
# # Writing the processed data
# data.to_csv("Processed_data.csv")
# print("Processed data successfully written")
