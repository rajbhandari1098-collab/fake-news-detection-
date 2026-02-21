import re, string, os, joblib, requests
from flask import Flask, render_template, request
from google import genai

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- MODELS ----
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
lr_model = joblib.load(os.path.join(BASE_DIR, "lr_model.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
gb_model = joblib.load(os.path.join(BASE_DIR, "gb_model.pkl"))
dt_model = joblib.load(os.path.join(BASE_DIR, "dt_model.pkl"))

client = genai.Client(api_key="ENTER YOUR GEMINI API ID.....")


def wordopt(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def gemini_check(news):
    r = client.models.generate_content(
        model="models/gemini-flash-lite-latest",
        contents=f"Fake or Real? Reply one word.\n\n{news}"
    )
    return "Not A Fake News" if "real" in r.text.lower() else "Fake News"

def gemini_explain(news):
    r = client.models.generate_content(
        model="models/gemini-flash-lite-latest",
        contents=f"Explain simply why this news is fake or real:\n\n{news}"
    )
    return r.text

# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/check", methods=["GET", "POST"])
def check():
    result = None
    explanation = None
    news_text = ""

    if request.method == "POST" and "clear" in request.form:
        return render_template(
        "check.html",
        result=None,
        explanation=None,
        news=""
    )


    if request.method == "POST":
        # 🔹 User input
        news_text = request.form.get("news", "")

        # 🔹 Text clean + vectorize
        clean = wordopt(news_text)
        vect = vectorizer.transform([clean])

        # 🔹 ML Predictions (0 / 1)
        lr_pred = lr_model.predict(vect)[0]
        rf_pred = rf_model.predict(vect)[0]
        gb_pred = gb_model.predict(vect)[0]
        dt_pred = dt_model.predict(vect)[0]

        # 🔹 Gemini Prediction
        gemini_result = gemini_check(news_text)

        # 🔹 Convert 0/1 → readable text
        lr = "Not A Fake News" if lr_pred == 1 else "Fake News"
        rf = "Not A Fake News" if rf_pred == 1 else "Fake News"
        gb = "Not A Fake News" if gb_pred == 1 else "Fake News"
        dt = "Not A Fake News" if dt_pred == 1 else "Fake News"

        # 🔹 Voting logic
        ml_votes = lr_pred + rf_pred + gb_pred + dt_pred
        ai_vote = 1 if gemini_result == "Not A Fake News" else 0
        total_votes = ml_votes + ai_vote   # out of 5

        final_result = "Not A Fake News" if total_votes >= 3 else "Fake News"
        confidence = round((total_votes / 5) * 100, 2)

        # 🔹 Result dict (NO 0/1 sent to HTML)
        result = {
            "LR": lr,
            "RF": rf,
            "GB": gb,
            "DT": dt,
            "GEMINI": gemini_result,
            "FINAL": final_result,
            "CONF": confidence
        }

        # 🔹 Gemini explanation (only when button clicked)
        if "explain" in request.form:
            explanation = gemini_explain(news_text)

    return render_template(
        "check.html",
        result=result,
        explanation=explanation,
        news=news_text
    )
@app.route("/about")
def about():
    return render_template("about.html")




if __name__ == "__main__":
    app.run(debug=True)