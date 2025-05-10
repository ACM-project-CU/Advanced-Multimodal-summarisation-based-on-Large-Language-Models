<<<<<<< HEAD
import fitz  # PyMuPDF
import torch
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# Function to compute ROUGE scores
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "ROUGE-1 F1": scores["rouge1"].fmeasure,
        "ROUGE-2 F1": scores["rouge2"].fmeasure,
        "ROUGE-L F1": scores["rougeL"].fmeasure,
    }

# Function to compute BERTScore
def compute_bert_score(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en")
    return {"Precision": P.item(), "Recall": R.item(), "F1": F1.item()}

# Function to compute METEOR Score
def compute_meteor(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return meteor_score(reference_tokens, candidate_tokens)

# Function to compute BLEU Score
def compute_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothing = SmoothingFunction().method4  # Using method4 smoothing for better results
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-gram to 4-gram
    return sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothing)

# Function to compute Perplexity Score (Fluency)
def compute_perplexity(text):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    encodings = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Function to normalize perplexity score to 0-1 range (higher is better)
def normalize_perplexity(perplexity, max_perplexity=1000):
    # Inverse and normalize perplexity (lower perplexity is better, so we invert)
    normalized = 1 - min(perplexity, max_perplexity) / max_perplexity
    return max(0, normalized)  # Ensure it's not negative

# Function to compute Cosine Similarity Score
def compute_cosine_similarity(reference, candidate):
    vectorizer = TfidfVectorizer().fit_transform([reference, candidate])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Function to compute Similarity Matrix Score
def compute_similarity_matrix(reference, candidate):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, candidate])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix[0, 1]  # Similarity score between original and summary

# Prompt user for PDF file paths
generated_summary_pdf_path = input("Enter the path to the generated summary PDF: ").strip()
standard_summary_pdf_path = input("Enter the path to the standard/reference summary PDF: ").strip()
original_pdf_path = input("Enter the path to the original text PDF (optional, press Enter to skip): ").strip()

# Extract text from PDFs
generated_summary = extract_text_from_pdf(generated_summary_pdf_path)
standard_summary = extract_text_from_pdf(standard_summary_pdf_path)
original_text = extract_text_from_pdf(original_pdf_path) if original_pdf_path else ""

# Check if text extraction was successful
if not generated_summary:
    print("Error: Could not extract text from the generated summary PDF.")
    exit(1)
if not standard_summary:
    print("Error: Could not extract text from the standard summary PDF.")
    exit(1)

# Compute Scores comparing generated summary with standard summary
rouge = compute_rouge(standard_summary, generated_summary)
bert = compute_bert_score(standard_summary, generated_summary)
meteor = compute_meteor(standard_summary, generated_summary)
bleu = compute_bleu(standard_summary, generated_summary)
perplexity_score = compute_perplexity(generated_summary)
normalized_perplexity = normalize_perplexity(perplexity_score)
cosine_sim = compute_cosine_similarity(standard_summary, generated_summary)
similarity_matrix_score = compute_similarity_matrix(standard_summary, generated_summary)

# Calculate a weighted ensemble score (higher is better)
ensemble_weights = {
    'rouge1': 0.15,
    'rouge2': 0.15,
    'rougeL': 0.15,
    'bert_f1': 0.15,
    'meteor': 0.10,
    'bleu': 0.10,
    'cosine': 0.10,
    'perplexity': 0.10
}

ensemble_score = (
    rouge["ROUGE-1 F1"] * ensemble_weights['rouge1'] +
    rouge["ROUGE-2 F1"] * ensemble_weights['rouge2'] +
    rouge["ROUGE-L F1"] * ensemble_weights['rougeL'] +
    bert["F1"] * ensemble_weights['bert_f1'] +
    meteor * ensemble_weights['meteor'] +
    bleu * ensemble_weights['bleu'] +
    cosine_sim * ensemble_weights['cosine'] +
    normalized_perplexity * ensemble_weights['perplexity']
)

# Print Results
print("\n--- Evaluation Scores (Generated Summary vs Standard Summary) ---")
print(f"ROUGE Scores:")
print(f"  ROUGE-1 F1: {rouge['ROUGE-1 F1']:.4f}")
print(f"  ROUGE-2 F1: {rouge['ROUGE-2 F1']:.4f}")
print(f"  ROUGE-L F1: {rouge['ROUGE-L F1']:.4f}")
print(f"BERTScore:")
print(f"  Precision: {bert['Precision']:.4f}")
print(f"  Recall: {bert['Recall']:.4f}")
print(f"  F1: {bert['F1']:.4f}")
print(f"METEOR Score: {meteor:.4f}")
print(f"BLEU Score: {bleu:.4f}")
print(f"Perplexity Score: {perplexity_score:.4f} (Lower is better)")
print(f"Normalized Perplexity: {normalized_perplexity:.4f} (Higher is better)")
print(f"Cosine Similarity Score: {cosine_sim:.4f} (Higher is better)")
print(f"Similarity Matrix Score: {similarity_matrix_score:.4f} (Higher is better)")
print(f"\n--- Overall Quality ---")
print(f"Ensemble Score: {ensemble_score:.4f} (Higher is better)")

# If original text was provided, also show comparison with original
if original_text:
    print("\n--- Additional Comparison with Original Text ---")
    orig_rouge = compute_rouge(original_text, generated_summary)
    orig_bert = compute_bert_score(original_text, generated_summary)
    orig_cosine = compute_cosine_similarity(original_text, generated_summary)
    print(f"ROUGE-L F1 (vs Original): {orig_rouge['ROUGE-L F1']:.4f}")
    print(f"BERTScore F1 (vs Original): {orig_bert['F1']:.4f}")
    print(f"Cosine Similarity (vs Original): {orig_cosine:.4f}")
=======
import fitz  # PyMuPDF
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# Function to compute ROUGE scores
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "ROUGE-1 F1": scores["rouge1"].fmeasure,
        "ROUGE-2 F1": scores["rouge2"].fmeasure,
        "ROUGE-L F1": scores["rougeL"].fmeasure,
    }

# Function to compute BERTScore
def compute_bert_score(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en")
    return {"Precision": P.item(), "Recall": R.item(), "F1": F1.item()}

# Function to compute METEOR Score
def compute_meteor(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return meteor_score(reference_tokens, candidate_tokens)

# Function to compute Perplexity Score (Fluency)
def compute_perplexity(text):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    encodings = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Function to compute Cosine Similarity Score
def compute_cosine_similarity(reference, candidate):
    vectorizer = TfidfVectorizer().fit_transform([reference, candidate])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Function to compute Similarity Matrix Score
def compute_similarity_matrix(reference, candidate):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, candidate])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix[0, 1]  # Similarity score between original and summary

# Prompt user for PDF file paths
original_pdf_path = input("Enter the path to the original text PDF: ").strip()
summary_pdf_path = input("Enter the path to the summarized text PDF: ").strip()

# Extract text from PDFs
original_text = extract_text_from_pdf(original_pdf_path)
generated_summary = extract_text_from_pdf(summary_pdf_path)

# Check if text extraction was successful
if not original_text:
    print("Error: Could not extract text from the original PDF.")
    exit(1)
if not generated_summary:
    print("Error: Could not extract text from the summary PDF.")
    exit(1)

# Compute Scores
rouge = compute_rouge(original_text, generated_summary)
bert = compute_bert_score(original_text, generated_summary)
meteor = compute_meteor(original_text, generated_summary)
perplexity_score = compute_perplexity(generated_summary)
cosine_sim = compute_cosine_similarity(original_text, generated_summary)
similarity_matrix_score = compute_similarity_matrix(original_text, generated_summary)

# Print Results
print("\n--- Evaluation Scores ---")
print(f"ROUGE Scores: {rouge}")
print(f"BERTScore: {bert}")
print(f"METEOR Score: {meteor:.4f}")
print(f"Perplexity Score: {perplexity_score:.4f} (Lower is better)")
print(f"Cosine Similarity Score: {cosine_sim:.4f} (Higher is better)")
print(f"Similarity Matrix Score: {similarity_matrix_score:.4f} (Higher is better)")
>>>>>>> 332449a7ae3ed30e52c34f172658560db2ded9b0
