import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB  # Using MultinomialNB as it is suitable with discrete features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import Processing_Data

# Reading the processed data

data = pd.read_csv("Processed_data.csv")


def encoding_category():
    label_encoder = LabelEncoder()
    data["Category_Encoded"] = data.Category
    data.Category_Encoded = label_encoder.fit_transform(data.Category_Encoded)


def train_test_split(column):
    skf = StratifiedKFold(shuffle=True)  # Using StratifiedKFold as the dataset is imbalanced such that training and
    # testing data is sampled with proportional ratio of classes
    for train_index, valid_index in skf.split(data.stemmed, data.Category_Encoded):
        X_train = data.loc[train_index, column]
        y_train = data.loc[train_index, "Category_Encoded"]
        X_valid = data.loc[valid_index, column]
        y_valid = data.loc[valid_index, "Category_Encoded"]
    return X_train, X_valid, y_train, y_valid


def naiveBayesClassifierPipeline():
    text_clf = Pipeline(steps=[
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultinomialNB(alpha=0.001))])  # Parameters tuned using GridSearchCV

    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_valid)
    print("F1 Score: ", f1_score(y_valid, predicted, average='weighted'))
    return text_clf


def svmClassifierPipeline():
    # {'clf-svm__alpha': 0.0001, 'clf-svm__loss': 'hinge', 'clf-svm__penalty': 'l1'}
    text_clf_svm = Pipeline(steps=[
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm', SGDClassifier(alpha=0.0001, loss='hinge', penalty='l1'))])  # Parameters tuned using GridSearchCV

    text_clf_svm = text_clf_svm.fit(X_train, y_train)
    predicted = text_clf_svm.predict(X_valid)
    print("F1 Score: ", f1_score(y_valid, predicted, average='weighted'))
    return text_clf_svm


def hyperParameterTuning(model):
    if model == "NB":
        param = {
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
        }
        gs_nb = GridSearchCV(text_clf, param_grid=param, n_jobs=-1, cv=5, verbose=5)
        gs_nb = gs_nb.fit(data.Stemmed_text, data.Category_Encoded)
        print(gs_nb.best_params_)
    elif model == "SVM":
        param = {
            "clf-svm__loss": ["hinge", "log"],
            "clf-svm__alpha": [0.0001, 0.001, 0.01, 0.1],
            "clf-svm__penalty": ["l2", "l1", "none", "elasticnet"],
        }
        gs_sgd = GridSearchCV(text_clf_svm, param_grid=param, n_jobs=-1, cv=5, verbose=5)
        gs_sgd = gs_sgd.fit(data.Stemmed_text, data.Category_Encoded)
        print(gs_sgd.best_params_)
    else:
        return 'Invalid model'


def urlClassifier(pipe):
    while True:
        url = input("Enter a URL: ")
        text = Processing_Data.url2text(url)
        stemmed_text = Processing_Data.preprocessingSingleText(text)
        stemmed_text_combined = " ".join(stemmed_text)
        prediction = pipe.predict(pd.Series([stemmed_text_combined] * 10))[0]
        print(data.loc[data.Category_Encoded == prediction, "Category"].iloc[0])
        cont = input("Enter Q to quit: ")
        if cont.lower() == 'q':
            break


# Encoding Category variable
encoding_category()

# Using stemmed text as input
data["stemmed"] = data.Stemmed_text.apply(lambda df: " ".join(df))

# Using lemmatized text as input
data["lemmatized"] = data.Lemmatized_text.apply(lambda df: " ".join(df))

# Splitting the dataset
X_train, X_valid, y_train, y_valid = train_test_split('Stemmed_text')
print(X_train, X_valid, y_train, y_valid)
# Training the dataset on Naive Bayes Classifier
text_clf = naiveBayesClassifierPipeline()

# Training the dataset on Naive Bayes Classifier
text_clf_svm = svmClassifierPipeline()

# hyperParameterTuning('NB')
# hyperParameterTuning('SVM')

urlClassifier(text_clf)  # Choose any of the pipelines to make predictions on a single url
