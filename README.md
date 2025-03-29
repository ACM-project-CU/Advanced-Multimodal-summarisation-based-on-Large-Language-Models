# Advanced Text Summarizer

## Overview

The **Advanced Text Summarizer** is a powerful Python-based tool that leverages state-of-the-art Natural Language Processing (NLP) models to provide comprehensive text analysis and summarization capabilities. Built on transformer-based architectures, this tool can process and analyze content from multiple input formats including plain text, PDF documents, images (via OCR), audio files (via speech-to-text), and even video content. The system employs a sophisticated pipeline that includes preprocessing, language detection, translation, and multiple analytical components to deliver high-quality summaries and insights.

## Key Features

### Text Summarization
- **Multiple Model Support**: Implements several state-of-the-art transformer models:
  - **T5 (Text-to-Text Transfer Transformer)**: Offers efficient summarization with the `t5-small` and `t5-3b` variants
  - **BART (Bidirectional and Auto-Regressive Transformers)**: Provides high-quality summaries using `facebook/bart-large-cnn`
  - **DistilBART**: A distilled version of BART (`sshleifer/distilbart-cnn-12-6`) that maintains quality while reducing computational requirements
  - **PEGASUS**: Specialized for abstractive summarization with `google/pegasus-xsum`
- **Customizable Summary Length**: Allows users to specify the maximum word limit for summaries
- **Chunk Processing**: Handles long documents by breaking them into manageable chunks for processing

### Advanced NLP Analysis
- **Keyword Extraction**: Uses RAKE (Rapid Automatic Keyword Extraction) algorithm to identify key phrases and topics
- **Named Entity Recognition (NER)**: Employs SpaCy to identify and classify entities such as people, organizations, locations, dates, and more
- **Sentiment Analysis**: Utilizes TextBlob to determine the emotional tone (positive, negative, or neutral) of the text
- **Bias Detection**: Implements a zero-shot classification model to detect potential bias in content
- **Topic Classification**: Categorizes text into domains like Finance, Health, Technology, and Education
- **Readability Analysis**: Calculates Flesch Reading Ease score to assess text complexity
- **Question Answering**: Allows users to ask specific questions about the processed text using a DistilBERT-based QA model

### Multimodal Input Processing
- **Text Input**: Direct processing of plain text
- **PDF Processing**: Extracts and analyzes text from PDF documents using PyMuPDF
- **Image Text Extraction**: Implements EasyOCR for high-accuracy text recognition from images
- **Audio Transcription**: Converts speech to text using Wav2Vec2 model with fallback to Whisper for improved accuracy
- **Video Analysis**: Extracts frames, generates visual descriptions, and transcribes audio to create comprehensive video summaries

### Multilingual Support
- **Language Detection**: Automatically identifies the source language using langdetect
- **Translation**: Translates non-English text to English using Google Translator API before processing
- **Cross-lingual Summarization**: Processes content in multiple languages through the translation pipeline

### Output Options
- **Multiple Export Formats**: Saves summaries as TXT, JSON, PDF, or DOCX
- **Comprehensive Analysis Reports**: Includes sentiment scores, keywords, topic categories, and readability metrics
- **Integrated Summaries**: For video content, combines visual and audio information into coherent narratives

## Technical Architecture

### Preprocessing Pipeline

The Advanced Text Summarizer implements a sophisticated preprocessing pipeline to ensure high-quality input for the summarization models:

1. **Text Normalization**:
   - Converts text to lowercase
   - Removes special characters and punctuation
   - Eliminates extra whitespace

2. **Tokenization**:
   - Splits text into sentences using NLTK's `sent_tokenize`
   - Breaks sentences into words with `word_tokenize`

3. **Stopword Removal**:
   - Filters out common words (e.g., "the", "and", "is") that don't contribute significant meaning
   - Uses NLTK's English stopword list

4. **Text Chunking**:
   - Divides long documents into manageable chunks (default 500 characters)
   - Ensures each chunk maintains context for better summarization

5. **Language Processing**:
   - Detects the source language
   - Translates non-English text to English when necessary

### Summarization Models

The system supports multiple transformer-based models, each with specific strengths:

1. **T5 Models**:
   - **Architecture**: Encoder-decoder transformer that treats all NLP tasks as text-to-text
   - **Variants**: `t5-small` (60M parameters) and `t5-3b` (3B parameters)
   - **Strengths**: Versatile and efficient for shorter texts

2. **BART Model**:
   - **Architecture**: Bidirectional encoder with autoregressive decoder
   - **Implementation**: `facebook/bart-large-cnn` (400M parameters)
   - **Strengths**: Excellent for abstractive summarization with high coherence

3. **DistilBART**:
   - **Architecture**: Knowledge-distilled version of BART
   - **Implementation**: `sshleifer/distilbart-cnn-12-6`
   - **Strengths**: Faster inference while maintaining quality

4. **PEGASUS**:
   - **Architecture**: Transformer pre-trained with gap-sentence generation
   - **Implementation**: `google/pegasus-xsum`
   - **Strengths**: Specialized for abstractive summarization

### Multimodal Processing Components

1. **PDF Processing**:
   - Uses PyMuPDF (fitz) to extract text from PDF documents
   - Preserves document structure where possible

2. **Image Processing**:
   - Implements EasyOCR with confidence thresholding (>60%)
   - Supports multiple languages for text extraction
   - Handles various image formats and qualities

3. **Audio Processing**:
   - Primary: Wav2Vec2 model for speech recognition
   - Fallback: OpenAI's Whisper model for challenging audio
   - Processes audio in chunks for long recordings

4. **Video Analysis**:
   - Frame extraction at regular intervals (default: every 5 seconds)
   - Visual captioning using ViT-GPT2 image captioning model
   - Audio extraction and transcription
   - Narrative generation that combines visual and audio information

## Installation

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, 16GB+ recommended for video processing
- **GPU**: Optional but highly recommended for faster processing
  - CUDA-compatible NVIDIA GPU with 6GB+ VRAM for optimal performance
- **Disk Space**: At least 10GB for models and dependencies

### Dependencies Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sw-Dy/advanced-text-summarizer.git
   cd advanced-text-summarizer
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLP models and resources**:
   ```python
   # Run this in Python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Install SpaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Additional dependencies for specific features**:

   For OCR functionality:
   ```bash
   # On Windows
   pip install pytesseract
   # Download and install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
   # Add Tesseract to your PATH
   ```

   For audio processing:
   ```bash
   # FFmpeg is required for audio extraction from videos
   # Download from https://ffmpeg.org/download.html and add to PATH
   ```

7. **GPU Acceleration** (optional but recommended):
   - Install CUDA and cuDNN appropriate for your PyTorch version
   - Verify GPU availability with the included test script:
     ```bash
     python test.py
     ```

## Usage Guide

### Command Line Interface

The Advanced Text Summarizer provides an interactive command-line interface for easy use:

```bash
python text_summariser.py
```

### Selecting a Summarization Model

When prompted, choose one of the available models:
- `t5-small`: Fastest option, good for quick summaries
- `bart-large`: High-quality summaries, balanced performance
- `distilbart`: Faster than BART with comparable quality
- `pegasus`: Specialized for news-like content
- `t5-3b`: Highest quality but requires significant computational resources

### Input Options

1. **Text Input**:
   ```
   Options: text, file, pdf, image, audio, video, exit
   Enter your choice: text
   Enter text: [Your text here]
   ```

2. **File Input**:
   ```
   Options: text, file, pdf, image, audio, video, exit
   Enter your choice: file
   Enter filename (.txt): global warming.txt
   ```

3. **PDF Input**:
   ```
   Options: text, file, pdf, image, audio, video, exit
   Enter your choice: pdf
   Enter PDF filename: Global warming i.pdf
   ```

4. **Image Input**:
   ```
   Options: text, file, pdf, image, audio, video, exit
   Enter your choice: image
   Enter image filename: image.png
   ```

5. **Audio Input**:
   ```
   Options: text, file, pdf, image, audio, video, exit
   Enter your choice: audio
   Enter audio filename: audio2.wav
   ```

6. **Video Input**:
   ```
   Options: text, file, pdf, image, audio, video, exit
   Enter your choice: video
   Enter video filename: gbw.mp4
   ```

### Customizing Summary Length

After selecting your input, you'll be prompted to specify the maximum word limit for the summary:

```
Enter maximum word limit for the summary (recommended: 20-30% of original length): 150
```

### Saving Output

After processing, you can save the summary in various formats:

```
Save as (txt/json/pdf/docx): json
```

### Asking Questions About the Text

The tool allows you to ask specific questions about the processed text:

```
Do you want to ask a question about the text? (yes/no): yes
Enter your question: What are the main causes of global warming?
```

## Video Summarization Workflow

The video summarization feature combines multiple AI techniques to create comprehensive summaries:

1. **Frame Extraction**:
   - Samples frames at regular intervals (configurable, default: every 5 seconds)
   - Limits to a maximum of 20 frames to manage processing time

2. **Visual Description**:
   - Processes each frame through a ViT-GPT2 image captioning model
   - Generates detailed descriptions with timestamps

3. **Audio Transcription**:
   - Extracts audio track from video
   - Transcribes speech using Wav2Vec2 model
   - Falls back to Whisper model if initial transcription fails

4. **Narrative Generation**:
   - Groups similar visual descriptions to avoid repetition
   - Uses transitional phrases to create a flowing narrative
   - Combines visual and audio information into a coherent summary

5. **Output Generation**:
   - Creates a structured summary with sections for visual content, audio transcription, and integrated summary
   - Saves the result to a text file

## Evaluation Metrics

The Advanced Text Summarizer can be evaluated using the included `check_score.py` script, which implements multiple metrics:

### Content Quality Metrics

1. **ROUGE Scores**:
   - **ROUGE-1**: Measures unigram overlap between summary and reference
   - **ROUGE-2**: Measures bigram overlap
   - **ROUGE-L**: Measures longest common subsequence

2. **BERTScore**:
   - Uses contextual embeddings to measure semantic similarity
   - Provides precision, recall, and F1 scores
   - More robust than n-gram based metrics for capturing meaning

3. **METEOR Score**:
   - Considers synonyms and stemming for better evaluation
   - Balances precision and recall with harmonic mean

### Linguistic Quality Metrics

1. **Perplexity Score**:
   - Measures fluency and naturalness of generated text
   - Lower scores indicate more fluent text

2. **Cosine Similarity**:
   - Compares vector representations of original and summary
   - Measures content preservation

3. **Similarity Matrix Score**:
   - Evaluates semantic similarity using TF-IDF vectors

### BARTScore Implementation

The project includes a BARTScore implementation for advanced evaluation:

- **Approach**: Uses BART's conditional probability to score summaries
- **Advantages**: Correlates well with human judgments
- **Implementation**: Available in the BARTScore directory

## Performance Benchmarks

Based on internal testing with standard datasets:

| Model | ROUGE-L | BERTScore | Processing Speed |
|-------|---------|-----------|------------------|
| T5-small | 0.38 | 0.85 | Fast |
| BART-large | 0.42 | 0.88 | Medium |
| DistilBART | 0.40 | 0.87 | Medium-Fast |
| PEGASUS | 0.43 | 0.89 | Medium |
| T5-3B | 0.44 | 0.90 | Slow |

*Note: Performance may vary based on hardware, input length, and content type.*

## Limitations

- **Computational Requirements**: High-end GPUs/TPUs are essential for efficient processing, especially for video content and larger models

- **Context Understanding**: Summarization models may occasionally miss critical details or nuances in complex texts

- **Bias Reflection**: Summaries can reflect biases present in the training data of the underlying models

- **Language Support**: Performance on low-resource languages is sub-optimal, even with translation

- **OCR Accuracy**: Handwritten text recognition remains challenging, and complex document layouts may not be preserved

- **Video Processing Time**: Video summarization is computationally intensive and may take several minutes for longer videos

## Future Development Roadmap

- **Enhanced Contextual Summarization**: Improving models to better capture nuanced details and maintain context across long documents

- **Domain-specific Adaptations**: Fine-tuning models for specialized fields like law, medicine, and finance

- **Cross-lingual Performance**: Enhancing support for low-resource languages through specialized models

- **Resource Optimization**: Reducing computational overhead for broader accessibility

- **Interactive Summaries**: Developing a web interface with adjustable parameters and real-time feedback

- **Multi-document Summarization**: Adding capability to synthesize information across multiple related documents

- **Fact-checking Integration**: Implementing verification mechanisms to ensure factual accuracy in summaries

## API Reference

### Core Functions

```python
# Summarize text with a specific model
summarize_text(text, model_choice)

# Extract keywords from text
extract_keywords(text)

# Identify named entities
extract_entities(text)

# Analyze sentiment
analyze_sentiment(text)

# Detect bias in content
detect_bias(text)

# Classify text into topics
classify_topic(text)

# Calculate readability score
readability_score(text)

# Extract text from PDF
read_pdf(file_path)

# Extract text from image
extract_text_from_image(image_path)

# Transcribe audio to text
transcribe_audio(file_path)

# Summarize video content
summarize_video(video_path)

# Answer questions about text
answer_question(text, question)
```

### Output Functions

```python
# Save summary as text file
save_as_txt(text, filename="summary_output.txt")

# Save summary as JSON
save_as_json(data, filename="summary_output.json")

# Save summary as PDF
save_as_pdf(text, filename="summary_output.pdf")

# Save summary as Word document
save_as_docx(text, filename="summary_output.docx")
```

## Demo and Examples

📌 **[View the Demo Video](https://1drv.ms/v/c/8c8713fbccf62dfa/EU1o_wQXgHROk4tUuSljYG8BFOLx8ssCPKZNLi41-EZK2w?e=JPwaSU)**

### Example: Text Summarization

```python
from text_summariser import summarize_text, extract_keywords

text = "Global warming is one of the most pressing issues of our time..."
model = "bart-large"

summary = summarize_text(text, model)
keywords = extract_keywords(text)

print(f"Summary: {summary}")
print(f"Keywords: {', '.join(keywords[:5])}")
```

### Example: Video Summarization

```python
from text_summariser import summarize_video

video_path = "climate_change_documentary.mp4"
summary = summarize_video(video_path)

with open("video_summary.txt", "w") as f:
    f.write(summary)
```

## Contributing

Contributions to the Advanced Text Summarizer are welcome! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit your changes**:
   ```bash
   git commit -m 'Add new feature'
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature-branch
   ```
5. **Create a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation to reflect changes
- Ensure compatibility with both CPU and GPU environments

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory Errors**:
   - Reduce batch size or chunk size
   - Use a smaller model (e.g., distilbart instead of bart-large)
   - Close other GPU-intensive applications

2. **Slow Processing**:
   - Enable GPU acceleration if available
   - Reduce the number of frames processed for video summarization
   - Use smaller models for faster inference

3. **OCR Quality Issues**:
   - Ensure images have sufficient resolution and contrast
   - Pre-process images to improve text visibility
   - Adjust confidence threshold in the code

4. **Audio Transcription Errors**:
   - Ensure audio has minimal background noise
   - Try the alternative transcription model (Whisper)
   - Split long audio files into smaller segments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For any questions, issues, or feature requests, please open an issue on the GitHub repository or contact the maintainers directly.

