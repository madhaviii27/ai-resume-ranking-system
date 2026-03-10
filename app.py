from flask import Flask, render_template, request
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


@app.route("/", methods=["GET", "POST"])
def index():

    results = []
    top_candidate = None

    if request.method == "POST":

        job_description = request.form["jobdesc"]
        files = request.files.getlist("resumes")

        resume_texts = []
        resume_names = []

        for file in files:

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            text = extract_text(filepath)

            resume_texts.append(text)
            resume_names.append(file.filename)

        documents = [job_description] + resume_texts

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_scores = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        for i, score in enumerate(similarity_scores):
            percent = round(score * 100, 2)
            results.append((resume_names[i], percent))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        if results:
            top_candidate = results[0]

    return render_template("index.html", results=results, top_candidate=top_candidate)


if __name__ == "__main__":
    app.run(debug=True)
