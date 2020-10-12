# Dependencies
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import pickle
import traceback
from Model import train_model, model_predict

# Your API definition
app = Flask(__name__, static_url_path="",
            static_folder="static", template_folder="templates")


@app.before_first_request
def _load_model():
    model = pickle.load(open('Voting_classifier/Model.pkl','rb'))


@app.route("/")
def hello():
    # return send_from_directory("static", filename="index.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    mail = (request.form["comment"])
    result = model_predict(mail)
    # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    return result


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 5000  # If you don"t provide any port the port will be set to 12345

    # serve efficiently a large model on a machine with many cores with many gunicorn workers, you can share the model parameters in memory using memory mapping

    app.run(port=port, debug=True)
