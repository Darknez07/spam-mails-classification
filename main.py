# Dependencies
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import pickle
import traceback
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


word_vectorizer = None
toxic_clf = None

# Your API definition
app = Flask(__name__, static_url_path="",
            static_folder="static", template_folder="templates")


@app.before_first_request
def _load_model():
    global word_vectorizer
    global toxic_clf

    with open("word_vectorizer.pkl", "rb") as vec:
        word_vectorizer = pickle.load(vec)

    with open("toxic_clf.pkl", "rb") as m_toxic:
        toxic_clf = pickle.load(m_toxic)


@app.route("/")
def hello():
    # return send_from_directory("static", filename="index.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    import pandas as pd
    # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    df = pd.read_table('SMSSpamCollection',
                       sep='\t',
                       header=None,
                       names=['label', 'sms_message'])

    df['label'] = df.label.map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                        df['label'],
                                                        random_state=1)

    comment = (request.form["comment"])

    X_test = pd.Series(
        [comment])
    print(X_test)
    y_test = pd.Series([1])

    # Instantiate the CountVectorizer method
    count_vector = CountVectorizer()
    # Fit the training data and then return the matrix
    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(X_test)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)

    predictions = naive_bayes.predict(testing_data)

    ans = "Not Spam" if precision_score(y_test, predictions) == 0 else "Spam"

    result = "<b>Your comment is: " + comment + "</b><hr>" + \
        " <b>Prediction: " + ans + "</b>"
    return result


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 5000  # If you don"t provide any port the port will be set to 12345

    # serve efficiently a large model on a machine with many cores with many gunicorn workers, you can share the model parameters in memory using memory mapping

    app.run(port=port, debug=True)
