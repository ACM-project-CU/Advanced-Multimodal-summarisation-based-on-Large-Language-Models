# Advanced Text Summarizer

## Overview

The **Advanced Text Summarizer** is a powerful Python-based tool that utilizes state-of-the-art NLP models for text summarization, keyword extraction, sentiment analysis, entity recognition, and more. It supports multiple input formats including text, PDF, images, and audio files.

## Features

- **Text Summarization:** Supports T5, BART, and DistilBART models.
- **Keyword Extraction:** Uses RAKE for extracting key phrases.
- **Named Entity Recognition (NER):** Identifies entities like names, locations, and organizations.
- **Sentiment Analysis:** Determines sentiment polarity of a given text.
- **Bias Detection:** Uses a classification model to detect bias in text.
- **Topic Classification:** Classifies text into categories such as Finance, Health, Technology, and Education.
- **Readability Analysis:** Calculates Flesch Reading Ease score.
- **Multilingual Support:** Detects and translates non-English text before processing.
- **Audio & Image Processing:** Converts speech to text and extracts text from images using OCR.
- **Multiple Output Formats:** Allows saving summaries as TXT, JSON, PDF, and DOCX.

## Installation

### Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Sw-Dy/advanced-text-summarizer.git
   cd advanced-text-summarizer
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download necessary NLP models:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
   ```bash
   python -m spacy download en_core_web_sm
   ```
5. (Optional) If using GPU, ensure you have **CUDA** installed for PyTorch acceleration.

## Usage

### Running the Application
```bash
python main.py
```

### Input Options
- **Text Input:** Directly enter text in the console.
- **File Input:** Load text from a `.txt` file.
- **PDF Input:** Extract and summarize text from PDFs.
- **Image Input:** Extract text using OCR and summarize it.
- **Audio Input:** Convert speech to text and summarize it.

### Choosing a Model
When prompted, select a summarization model:
- `t5-small`
- `bart-large`
- `distilbart`

## Workflow

The **Advanced Text Summarizer** follows a structured pipeline to process and summarize text from various sources. Below is the step-by-step workflow:

### 1. Input Processing
The tool accepts multiple input formats:
- **Raw Text:** Direct text input from the user.
- **PDF Files:** Extracts text using PyMuPDF.
- **Images:** Uses Optical Character Recognition (OCR) to extract text.
- **Audio Files:** Converts speech to text using a Speech-to-Text model.

### 2. Preprocessing
- **Tokenization & Cleaning:** Splits text into meaningful components.
- **Language Detection:** Identifies the language of the input text.
- **Translation (if needed):** Converts non-English text to English for uniform processing.
- **Stopword Removal:** Eliminates common words that do not add value to the summary.

### 3. Summarization & NLP Analysis
Once preprocessed, the text undergoes advanced NLP-based analysis:
- **Text Summarization:** Uses state-of-the-art models such as **T5, BART, and DistilBART** to generate concise summaries.
- **Sentiment Analysis:** Detects the emotional tone of the text.
- **Keyword Extraction:** Identifies important terms using **RAKE**.
- **Named Entity Recognition (NER):** Detects entities such as names, locations, and organizations.
- **Bias Detection:** Identifies potential biases in the text.
- **Topic Classification:** Categorizes text into domains like **Finance, Health, Technology, Education, etc.**
- **Readability Analysis:** Computes the **Flesch Reading Ease** score.

### 4. Output Generation
After analysis, the tool generates structured outputs:
- **Formats:** TXT, JSON, PDF, DOCX.
- **Insights:** Includes sentiment scores, keywords, topic categories, and readability analysis.

---

📌 **[View the Workflow Diagram](dia2.png)**  
_(Replace `#` with the actual link to the workflow image)_


## Evaluation
This summarizer achieves a **high BERTScore**, ensuring high-quality summaries that preserve meaning and coherence.


**Limitations:** 

-Computational cost – High end GPUs/TPUs is essential for efficient processing 

-Context understanding – Summarization models may miss some critical details 

-Bias – Summaries can reflect biases from training data 

-Language limitations – Performance on low-resource languages is sub-optimal 

-OCR accuracy – Handwritten text recognition remains to be a challenge  

**Future Scope**

-Enhancing contextual summarization – Improving the models to capture nuanced details 

-Domain-specific adaptations – It can be fine tuned for specialized fields like law, medicine, and finance 

-Cross-lingual performance – Support can be enhanced for the low-resource languages 

-Optimized resource utilization – Reduction in the computational overheads for broader accessibility 


📌 **[View the Demo Video](https://1drv.ms/v/c/8c8713fbccf62dfa/EU1o_wQXgHROk4tUuSljYG8BFOLx8ssCPKZNLi41-EZK2w?e=JPwaSU)** 

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## License
This project is licensed under the MIT License.

---

For any queries, feel free to open an issue or contribute to the project!

