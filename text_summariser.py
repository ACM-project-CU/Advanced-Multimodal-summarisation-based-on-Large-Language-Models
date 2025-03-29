import re
import json
import logging
import nltk
from transformers import pipeline
from textwrap import wrap
from langdetect import detect
from nltk.corpus import stopwords
from textblob import TextBlob
from deep_translator import GoogleTranslator
from nltk.tokenize import sent_tokenize, word_tokenize
from rake_nltk import Rake
import spacy
import PyPDF2
import pytesseract
from PIL import Image
import whisper
from fpdf import FPDF
from docx import Document
import textstat
import easyocr 
import speech_recognition as sr
import torch
import openai
import os
from pydub import AudioSegment
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import cv2
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


# Initialize necessary components
logging.basicConfig(filename="summarizer.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
models = {
    "t5-small": "t5-small",
    "bart-large": "facebook/bart-large-cnn",
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "pegasus": "google/pegasus-xsum",
    "t5-3b": "t5-3b"
}
stopwords.words("english")  # Download if not already done
nlp = spacy.load("en_core_web_sm")

# Preprocessing Functions
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Allow only alphanumeric and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words("english") and len(word) > 2]

    return " ".join(words)

def chunk_text(text, chunk_size=500):
    return wrap(text, width=chunk_size)

# Summarization Functions
def summarize_text(text, model_choice):
    try:
        # Get the maximum word limit from user input
        max_words = int(input("Enter maximum word limit for the summary (recommended: 20-30% of original length): "))
        
        # Initialize the summarization model
        model_name = models.get(model_choice, "facebook/bart-large-cnn")
        model = pipeline("summarization", model=model_name, tokenizer=model_name, framework="pt", device=0 if torch.cuda.is_available() else -1)

        
        # Split text into manageable chunks and summarize each chunk
        text_chunks = chunk_text(text)
        summaries = []
        for chunk in text_chunks:
            try:
                summary_output = model(chunk, max_length=max_words, min_length=10, do_sample=False)
                summaries.append(summary_output[0]['summary_text'])
            except Exception as e:
                logging.error(f"Summarization error: {e}")
                summaries.append(chunk)  # Fallback to original chunk if summarization fails


        
        # Combine summaries and truncate to match the max word limit
        final_summary = " ".join(summaries)
        final_summary_words = final_summary.split()
        if len(final_summary_words) > max_words:
            final_summary = " ".join(final_summary_words[:max_words])
        
        return final_summary
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return "Error in summarization. Check logs."


# Additional Features
def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def detect_bias(text):
    classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
    labels = ["biased", "neutral"]
    result = classifier(text, candidate_labels=labels)
    return result

def classify_topic(text):
    classifier = pipeline("zero-shot-classification")
    labels = ["Finance", "Health", "Technology", "Education"]
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0]

def compression_ratio(original, summary):
    return (1 - len(summary) / len(original)) * 100

def readability_score(text):
    return round(textstat.flesch_reading_ease(text), 2)  # Round for consistency


# File Handling Functions
def read_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        logging.error(f"PDF reading error: {e}")
        return "Error reading PDF file."

import easyocr
import logging
import os

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

def extract_text_from_image(image_path):
    try:
        # Initialize EasyOCR reader (English language, using GPU if available)
        reader = easyocr.Reader(['en'], gpu=True)

        # Perform OCR
        results = reader.readtext(image_path, detail=1)  # Returns text along with confidence

        
        

        extracted_text = []

        for entry in results:
            if len(entry) >= 2:  # Ensure structure contains text and confidence
                text, confidence = entry[1], entry[2]
                
                # Apply confidence threshold (e.g., 60%)
                if confidence > 0.6:
                    extracted_text.append(text)

        final_text = "\n".join(extracted_text)  # Join text with new lines

        return final_text if final_text.strip() else "No text detected."

    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file using Wav2Vec2 model for speech recognition.

    :param file_path: Path to the audio file (WAV or other formats like MP3).
    :return: Transcribed text as a string or an error message.
    """
    try:
        if not os.path.exists(file_path):
            return "Error: Audio file not found. Please check the file path."

        # Load pre-trained model and tokenizer
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Load audio
        sample_rate = 16000
        speech, rate = librosa.load(file_path, sr=sample_rate)
        
        # Define chunk size (30 seconds)
        chunk_duration = 30
        samples_per_chunk = chunk_duration * sample_rate

        # Process and transcribe each chunk
        transcriptions = []
        for start_idx in range(0, len(speech), samples_per_chunk):
            chunk = speech[start_idx : start_idx + samples_per_chunk]
            input_values = tokenizer(chunk, return_tensors='pt').input_values
            logits = model(input_values).logits

            # Decode predicted token IDs
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])
            transcriptions.append(transcription)

        full_transcription = " ".join(transcriptions)
        return full_transcription
    
    except Exception as e:
        return f"Error during transcription: {str(e)}"
    
def summarize_video(video_path: str) -> str:
    """
    Analyze and summarize video content by extracting frames, generating descriptions,
    and transcribing audio to create a comprehensive summary.
    
    :param video_path: Path to the video file.
    :return: A descriptive summary of the video content or an error message.
    """
    try:
        if not os.path.exists(video_path):
            return "Error: Video file not found. Please check the file path."
            
        # Load pre-trained vision-language model for image captioning
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        # Set device to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file."
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Calculate frame sampling rate (extract frames every N seconds)
        sample_interval = 5  # Extract a frame every 5 seconds
        frames_to_sample = int(duration / sample_interval)
        frames_to_sample = min(frames_to_sample, 20)  # Limit to 20 frames maximum
        
        if frames_to_sample <= 0:
            frames_to_sample = 1  # Ensure at least one frame is processed
            
        frame_interval = max(1, int(frame_count / frames_to_sample))
        
        # Extract frames and generate captions
        frame_descriptions = []
        current_frame = 0
        
        print(f"Analyzing video... (extracting {frames_to_sample} frames)")
        
        while current_frame < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB (OpenCV uses BGR by default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare image for the model
            pixel_values = feature_extractor(images=rgb_frame, return_tensors="pt").pixel_values.to(device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Add timestamp information
            timestamp = current_frame / fps
            minutes, seconds = divmod(int(timestamp), 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            
            frame_descriptions.append(f"[{timestamp_str}] {caption}")
            current_frame += frame_interval
        
        # Release video capture
        cap.release()
        
        # Extract audio from video and transcribe it
        print("Extracting and transcribing audio from video...")
        
        # Create a temporary audio file
        temp_audio_path = os.path.splitext(video_path)[0] + "_temp_audio.wav"
        
        try:
            # Use moviepy to extract audio from video
            from moviepy.editor import VideoFileClip # type: ignore
            
            # Extract audio using moviepy
            print("Extracting audio using moviepy...")
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(temp_audio_path, logger=None)
            audio_clip.close()
            video_clip.close()
            
            # Transcribe the extracted audio using Wav2Vec2
            audio_transcription = transcribe_audio(temp_audio_path)
            
            # If Wav2Vec2 transcription fails or returns an error, try using Whisper as a fallback
            if audio_transcription.startswith("Error") or not audio_transcription.strip():
                print("Trying alternative transcription method with Whisper...")
                try:
                    # Load Whisper model
                    whisper_model = whisper.load_model("base")
                    
                    # Transcribe with Whisper
                    result = whisper_model.transcribe(temp_audio_path)
                    audio_transcription = result["text"]
                except Exception as whisper_error:
                    logging.error(f"Whisper transcription error: {whisper_error}")
                    # Keep the original error message if Whisper also fails
            
            # Clean up the temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        except Exception as e:
            logging.error(f"Audio extraction error: {e}")
            audio_transcription = "Could not extract or transcribe audio from the video."
            
            # Clean up the temporary audio file if it exists
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        # Combine frame descriptions into a coherent summary
        if not frame_descriptions:
            return "Could not extract any meaningful content from the video."
            
        # Generate a summary from the frame descriptions
        descriptions_text = "\n".join(frame_descriptions)
        
        # Use the existing summarization function to create a concise summary
        print("\nFrame-by-frame descriptions:")
        print(descriptions_text)
        
        print("\nGenerating final video summary...")
        
        # Create a summary that captures the essence of the video
        summary = f"Video Summary ({int(duration)} seconds):\n\n"
        
        # Extract captions without timestamps
        captions = [desc.split('] ')[1] for desc in frame_descriptions]
        
        # Group similar captions to avoid repetition
        grouped_captions = []
        current_group = [captions[0]]
        
        for i in range(1, len(captions)):
            # Simple similarity check - if captions share significant words
            current_words = set(current_group[-1].lower().split())
            next_words = set(captions[i].lower().split())
            common_words = current_words.intersection(next_words)
            
            # If there's significant overlap, group them
            if len(common_words) >= 2 and len(common_words) / len(next_words) > 0.3:
                current_group.append(captions[i])
            else:
                # Process the current group before starting a new one
                if current_group:
                    grouped_captions.append(current_group)
                current_group = [captions[i]]
        
        # Add the last group
        if current_group:
            grouped_captions.append(current_group)
        
        # Generate coherent paragraphs from grouped captions
        paragraphs = []
        
        # Transitional phrases to make the summary more natural
        transitions = [
            "The video begins with", "Initially, the video shows", "At the start,", 
            "The scene then changes to", "Subsequently,", "Following this,", 
            "Next, we can see", "The video continues with", "Later,", 
            "Towards the middle,", "As the video progresses,", 
            "The focus then shifts to", "Afterward,", 
            "Towards the end,", "Finally,", "The video concludes with"
        ]
        
        for i, group in enumerate(grouped_captions):
            # Choose an appropriate transition based on position in the video
            if i == 0:
                transition = transitions[0]
            elif i == len(grouped_captions) - 1:
                transition = transitions[-1]
            else:
                # Select a middle transition based on position
                idx = min(i + 3, len(transitions) - 3)
                transition = transitions[idx]
            
            # Combine similar captions into a coherent sentence
            if len(group) == 1:
                paragraph = f"{transition} {group[0]}"
            else:
                # Extract key elements from similar captions
                combined = group[0]
                for caption in group[1:]:
                    # Find unique elements in this caption
                    current_words = set(combined.lower().split())
                    new_words = set(caption.lower().split())
                    unique_words = new_words - current_words
                    
                    if unique_words:
                        # Add unique elements to the combined description
                        unique_phrase = ' '.join([w for w in caption.split() if w.lower() in unique_words])
                        combined += f" with {unique_phrase}"
                
                paragraph = f"{transition} {combined}"
            
            paragraphs.append(paragraph)
        
        # Combine paragraphs into a flowing narrative
        narrative = ". ".join(paragraphs) + "."
        
        # Clean up any double periods or spacing issues
        narrative = narrative.replace("..", ".").replace(" .", ".").replace(".", ". ").strip()
        
        summary += narrative
        
        # Add audio transcription to the summary if available
        if 'audio_transcription' in locals() and audio_transcription and not audio_transcription.startswith("Error") and not audio_transcription.startswith("Could not"):
            # Process the transcription to make it more readable
            processed_transcription = audio_transcription.strip()
            print("\nAudio Transcription:")
            print(processed_transcription)
            
            # Add the transcription section to the summary
            summary += "\n\nAudio Transcription:\n\n"
            
            # If the transcription is very long, summarize it
            if len(processed_transcription.split()) > 200:
                # Use the existing summarization function to create a concise summary of the transcription
                print("Summarizing audio transcription...")
                try:
                    # Initialize a summarization model
                    model_name = "facebook/bart-large-cnn"
                    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name, framework="pt", device=0 if torch.cuda.is_available() else -1)
                    
                    # Split text into manageable chunks and summarize each chunk
                    text_chunks = chunk_text(processed_transcription)
                    transcription_summaries = []
                    
                    for chunk in text_chunks:
                        if len(chunk.split()) > 10:  # Only summarize if there's enough content
                            summary_output = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                            transcription_summaries.append(summary_output[0]['summary_text'])
                        else:
                            transcription_summaries.append(chunk)
                    
                    # Combine summaries
                    processed_transcription = " ".join(transcription_summaries)
                except Exception as e:
                    logging.error(f"Transcription summarization error: {e}")
                    # If summarization fails, use the original transcription
                    pass
            
            summary += processed_transcription
            
            # Create a combined summary section that integrates visual and audio information
            summary += "\n\nIntegrated Summary:\n\n"
            
            # Create a more coherent narrative by combining visual and audio information
            try:
                # Initialize a summarization model for the final integration
                model_name = "facebook/bart-large-cnn"
                integrator = pipeline("summarization", model=model_name, tokenizer=model_name, framework="pt", device=0 if torch.cuda.is_available() else -1)
                
                # Combine visual narrative and processed transcription
                combined_text = f"Visual content: {narrative} Audio content: {processed_transcription}"
                
                # Generate an integrated summary
                integrated_summary = integrator(combined_text, max_length=150, min_length=50, do_sample=False)
                summary += integrated_summary[0]['summary_text']
            except Exception as e:
                logging.error(f"Integration summarization error: {e}")
                # If integration fails, provide a simple combined summary
                summary += "This video contains visual elements showing " + narrative.lower() + " The audio discusses " + processed_transcription[:100] + "..."
        
        return summary
        
    except Exception as e:
        logging.error(f"Video summarization error: {e}")
        return f"Error during video summarization: {str(e)}"
    
def answer_question(text, ask):
    """
    Answer questions based on the given text using a pre-trained QA model.
    
    :param text: The original document content.
    :param question: The question asked by the user.
    :return: Answer extracted from the text.
    """
    try:
        qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa_model(question=ask, context=text)
        return result["answer"]
    except Exception as e:
        return f"Error processing question: {str(e)}"


# Output Functions
def save_as_txt(text, filename="summary_output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def save_as_json(data, filename="summary_output.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_as_pdf(text, filename="summary_output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)

def save_as_docx(text, filename="summary_output.docx"):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)

# Main Application
def main():
    print("Welcome to the Advanced Text Summarizer!")
    print("Choose a summarization model (t5-small/bart-large/distilbart/pegasus/t5-3b):")
    model_choice = input("Model: ")
    user_input = ""

    while True:
        print("\nOptions: text, file, pdf, image, audio, video, exit")
        choice = input("Enter your choice: ").lower()

        if choice == 'exit':
            print("Exiting. Goodbye!")
            break

        elif choice == 'text':
            user_input = input("Enter text: ")
        
        elif choice == 'file':
            filename = input("Enter filename (.txt): ")
            with open(filename, "r", encoding="utf-8") as f:
                user_input = f.read()

        elif choice == 'pdf':
            filename = input("Enter PDF filename: ")
            user_input = read_pdf(filename)

        elif choice == 'image':
            filename = input("Enter image filename: ")
            user_input = extract_text_from_image(filename)
            if "Error" in user_input:
                print(user_input)
                continue


        elif choice == 'audio':
            filename = input("Enter audio filename: ")
            user_input = transcribe_audio(filename)
            
        elif choice == 'video':
            filename = input("Enter video filename: ")
            user_input = summarize_video(filename)
            if "Error" in user_input:
                print(user_input)
                continue
        
        else:
            print("Invalid choice.")
            continue

        if not user_input.strip():
            print("Empty input! Try again.")
            continue

        language = detect(user_input)
        if language != "en":
            print(f"Detected language: {language}, translating to English...")
            user_input = GoogleTranslator(source=language, target="en").translate(user_input)


        processed_text = preprocess_text(user_input)
        summary = summarize_text(processed_text, model_choice)


        print("\nSummary:")
        print(summary)

        save_format = input("Save as (txt/json/pdf/docx): ").lower()
        if save_format == 'txt':
            save_as_txt(summary)
        elif save_format == 'json':
            save_as_json({"summary": summary})
        elif save_format == 'pdf':
            save_as_pdf(summary)
        elif save_format == 'docx':
            save_as_docx(summary)
        else:
            print("Invalid save format.")
        
        while True:
            ask_choice = input("Do you want to ask a question about the text? (yes/no): ").lower()
            if ask_choice == 'yes':
                ask = input("Enter your question: ")
                answer = answer_question(user_input, ask)
                print("\nAnswer:\n", answer)
            elif ask_choice == 'no':
                break
            else:
                print("Invalid input! Please type 'yes' or 'no'.")

if __name__ == "__main__":
    main()
