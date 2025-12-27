import os
#  flask is an app object, render template is to load the html file , request =to fetch the file , response = to stream the data , send file is to download the file 
from flask import Flask, render_template, request, Response, send_file 
# to extract texts from the pdfs
from PyPDF2 import PdfReader
# to import groq llm
from groq import Groq
#
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
SUMMARY_FOLDER = "summaries"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

latest_summary = ""

# ================= PDF TEXT EXTRACTION =================
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ================= STREAM SUMMARY =================
def stream_summary(text):
    global latest_summary
    latest_summary = ""

    prompt = f"""
Summarize the following document clearly.
Use short paragraphs or bullet points.

{text[:12000]}
"""

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            latest_summary += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content

# ================= SAVE SUMMARY PDF =================
def create_summary_pdf(text):
    file_path = os.path.join(SUMMARY_FOLDER, "summary.pdf")
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = [Paragraph(text.replace("\n", "<br/>"), styles["Normal"])]
    doc.build(content)
    return file_path

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    print("Summarize route called")

    if "pdf" not in request.files:
        return "No PDF uploaded", 400

    file = request.files["pdf"]
    print("Received file:", file.filename)

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text_from_pdf(path)
    print("Extracted text length:", len(text))

    return Response(stream_summary(text), mimetype="text/plain")

@app.route("/download")
def download():
    pdf_path = create_summary_pdf(latest_summary)
    return send_file(pdf_path, as_attachment=True)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
