from flask import Flask, render_template, request, jsonify
import pickle

cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get("email-content")
    tokenized_email = cv.transform([email])
    predictions = clf.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions=predictions, email_text=email)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    email = data["content"]
    tokenized_email = cv.transform([email])
    predictions = clf.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return jsonify({"predictions":predictions, "email":email})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)