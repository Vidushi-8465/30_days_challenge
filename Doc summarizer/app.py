import os
from PyPDF2 import PdfReader
from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
SUMMARY_FOLDER = "summaries"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

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
def stream_summary(text, length):
    global latest_summary
    latest_summary = ""

    if length == "short":
        max_tokens = 400
    elif length == "medium":
        max_tokens = 800
    else:
        max_tokens = 1500

    prompt = f"""
Summarize the following document clearly.
Use short paragraphs or bullet points.

{text[:12000]}
"""

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            latest_summary += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content

# ================= SAVE SUMMARY PDF =================
def create_summary_pdf(text):
    path = os.path.join(SUMMARY_FOLDER, "summary.pdf")
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    content = [Paragraph(text.replace("\n", "<br/>"), styles["Normal"])]
    doc.build(content)
    return path

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if "pdf" not in request.files:
        return "No file uploaded", 400

    pdf = request.files["pdf"]
    length = request.form.get("length", "medium")

    file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(file_path)

    text = extract_text_from_pdf(file_path)

    return Response(
        stream_summary(text, length),
        mimetype="text/plain"
    )

@app.route("/download")
def download():
    pdf_path = create_summary_pdf(latest_summary)
    return send_file(pdf_path, as_attachment=True)

