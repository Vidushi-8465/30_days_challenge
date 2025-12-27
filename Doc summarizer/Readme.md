---

##  Day 6 (Project 3): Document Summarizer Web App using LLM (Groq API)

This project is a **web-based document summarizer** that allows users to upload a **PDF file** and get a **concise AI-generated summary** using a **Large Language Model (LLM)** powered by **Groq**.

The summary is streamed in real time and can also be **downloaded as a PDF**.

---

##  Overview

Reading long documents is time-consuming.
This application automates the process by:

• Uploading a PDF
• Extracting text from it
• Sending the text to an LLM (Groq)
• Generating a summarized version
• Streaming the summary live
• Allowing the user to download the summary as a PDF

---

##  Tech Stack Used

### Frontend

• HTML
• CSS
• JavaScript (Fetch API + Streaming)

### Backend

• Python
• Flask (Web framework)
• Groq API (LLM inference)

### Libraries

• `PyPDF2` – PDF text extraction
• `fpdf` – PDF generation for summary
• `Flask` – backend routing and API handling

---

##  Project Structure

```
document-summarizer/
│
├── app.py                # Flask backend
├── templates/
│   └── index.html        # Frontend UI
│    uploads              # uploaded files
│   └──mod 3 ese
│    summaries            # summaries downloaded
│   └──pdf
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ How It Works (Flow)

1. User uploads a PDF from the browser
2. PDF is sent to the Flask backend
3. Text is extracted using PyPDF2
4. Extracted text is sent to **Groq LLM API**
5. Summary is generated **token by token (streaming)**
6. Summary is displayed live on the webpage
7. User can download the summary as a PDF

---

##  API Used

### Groq LLM API

• Used instead of OpenAI
• Faster inference
• Free-tier friendly
• Supports streaming responses

Model example:

```
llama3-8b-8192
```

---

##  Features

• Upload PDF files
• AI-powered summarization
• Real-time streaming summary
• Clean web UI
• Download summary as PDF
• No database required

---

##  Limitations

1. Very large PDFs may take longer to summarize
2. Summary quality depends on document clarity
3. Token limits apply based on Groq model
4. Requires internet connection for API calls

---

##  Future Improvements

• Add word-length control for summary
• Support DOCX and TXT files
• Add authentication
• Add multiple summary styles (bullet / paragraph)
• Deploy on cloud (Render / Railway / Vercel)

---

##  Key Learnings from this Project

• How to integrate an LLM API in a web app
• How to stream AI responses in real time
• Handling file uploads in Flask
• Converting AI output into downloadable PDFs
• Replacing OpenAI with Groq API seamlessly

---

