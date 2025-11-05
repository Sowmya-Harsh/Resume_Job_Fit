import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import re
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Constants ===
MODEL_PATH = "mlp_resume_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
SBERT_MODEL = "all-MiniLM-L6-v2"
LABEL_MAP = {0: "Good Fit", 1: "No Fit", 2: "Potential Fit"}

# === Load Model & Tools ===
model = joblib.load(MODEL_PATH)
sbert = SentenceTransformer(SBERT_MODEL)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

# === Feature Functions ===
def extract_years_experience(text):
    matches = re.findall(r'(\d+)(\+)?\s*(?:years?|yrs?)', str(text).lower())
    return int(matches[0][0]) if matches else 0

def title_match(resume, job_description):
    resume = str(resume).lower()
    job_title_words = job_description.lower().split()[:5]
    return int(any(word in resume for word in job_title_words))

def skill_match_ratio(resume, job_description):
    job_words = set(re.findall(r'\b\w+\b', job_description.lower()))
    return sum(1 for word in job_words if word in resume.lower()) / max(len(job_words), 1)

def jaccard_sim(text1, text2):
    s1 = set(re.findall(r'\b\w+\b', str(text1).lower()))
    s2 = set(re.findall(r'\b\w+\b', str(text2).lower()))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

def generate_features(resume, job):
    resume_emb = sbert.encode([resume])[0]
    job_emb = sbert.encode([job])[0]
    
    skill_ratio = skill_match_ratio(resume, job)
    exp_diff = abs(extract_years_experience(resume) - extract_years_experience(job))
    title_flag = title_match(resume, job)
    tfidf_sim = cosine_similarity(tfidf_vectorizer.transform([resume]), tfidf_vectorizer.transform([job]))[0][0]
    jaccard = jaccard_sim(resume, job)

    structured = np.array([skill_ratio, exp_diff, title_flag, tfidf_sim, jaccard])
    return np.hstack([resume_emb, job_emb, structured]).reshape(1, -1)

# === PDF Text Extraction ===
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img) + "\n"
    return text.strip()

# === Unified Prediction Function ===
def predict_fit(resume_pdf, resume_text, job_pdf, job_text):
    resume_input = resume_text.strip() if resume_text.strip() else (
        extract_text_from_pdf(resume_pdf) if resume_pdf else None)
    job_input = job_text.strip() if job_text.strip() else (
        extract_text_from_pdf(job_pdf) if job_pdf else None)

    if not resume_input or not job_input:
        return " Please provide both resume and job description, either via text or PDF."

    features = generate_features(resume_input, job_input)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features).max()
    return f" Prediction: {LABEL_MAP[pred]} (Confidence: {proba:.2%})"

# === Gradio Interface ===
iface = gr.Interface(
    fn=predict_fit,
    inputs=[
        gr.File(label="Upload Resume (PDF)", type="binary"),
        gr.Textbox(label="Resume Text (optional, overrides PDF)", lines=10, placeholder="Paste resume text here..."),
        gr.File(label="Upload Job Description (PDF)", type="binary"),
        gr.Textbox(label="Job Description Text (optional, overrides PDF)", lines=10, placeholder="Paste job description here...")
    ],
    outputs="text",
    title="Resume-Job Fit Classifier",
    description="Upload or paste a resume and job description to assess fit (using SBERT + MLP + structured features)."
)

if __name__ == "__main__":
    iface.launch()
