import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import  VotingClassifier
import  pickle as pck
import os

def get_data():
    df = pd.read_table('SMSSpamCollection',
                    sep='\t',
                    header=None,
                    names=['label', 'sms_message'])

    df['label'] = df.label.map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1,
                                                    test_size=0.1)

    count_vector = text.CountVectorizer(ngram_range=[1, 4],
                                        analyzer='char_wb')

    # Fit the training data and then return the matrix

    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(X_test)

    # NEw layer of tf-idf for better model

    transformer = text.TfidfTransformer()
    training_data = transformer.fit_transform(training_data)
    testing_data = transformer.transform(testing_data)

    return (training_data,y_train), (testing_data, y_test), count_vector, transformer

def train_model():

    (training_data, y_train),(testing_data, y_test), _, _ = get_data()
    # Naive Bayes
    naive_bayes = MultinomialNB()

    # SVC
    svc = SVC(kernel='linear')

    # Gaussian NB
    gnb = GaussianNB()

    vcc  =VotingClassifier(estimators = [('m1',naive_bayes),('m2',svc),('m3',gnb)],
                           voting='hard',
                           n_jobs=-1)

    vcc.fit(training_data.toarray(), y_train)
    if not os.path.exists('Voting_classifier'):
        os.makedirs('Voting_classifier')
    pck.dump(vcc, open('Voting_classifier/Model.pkl','wb'))
    print(classification_report(y_test,vcc.predict(testing_data.toarray()),target_names=["Ham","Spam"]))

def model_predict(mail):
    data = pd.DataFrame([mail])
    print(data)
    _, _, counter, transformer = get_data()

    # Convert data

    first = counter.transform(data[0])
    second = transformer.transform(first)

    # Get predictions
    model = pck.load(open('Voting_classifier/Model.pkl','rb'))

    ans = "Not Spam" if model.predict(second.toarray())[0] == 0 else "Spam"

    result = "<b>Your mail is: " + mail + "</b><hr>" + \
    " <b>Prediction: " + ans + "</b>"

    return result


# Uncomment this after unextracting the Model.rar
# If you run this script you will get the result on the test data
# (training_data, y_train),(testing_data, y_test), _, _ = get_data()
# model = pck.load(open('Voting_classifier/Model.pkl','rb'))
# print(classification_report(y_test,
#                             model.predict(np.array(testing_data.toarray(),dtype=np.float64)),
#                             target_names=["Ham","Spam"]))
# print()
print(model_predict('Hey buddy how are you'))