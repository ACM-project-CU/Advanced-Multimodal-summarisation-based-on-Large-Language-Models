import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.utils import secure_filename
from text_summariser import (
    summarize_text, preprocess_text, extract_keywords, extract_entities,
    analyze_sentiment, classify_topic, readability_score, read_pdf,
    extract_text_from_image, transcribe_audio, summarize_video, answer_question,
    save_as_txt, save_as_json, save_as_pdf, save_as_docx
)
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure results folder
RESULTS_FOLDER = 'results'
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Store session data
session_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Get model choice
        model_choice = request.form.get('model_choice', 'bart-large')
        
        # Get max words
        max_words = int(request.form.get('max_words', 150))
        
        # Determine input type and process accordingly
        user_input = ""
        input_type = "text"
        
        if request.form.get('text_input'):
            user_input = request.form.get('text_input')
            input_type = "text"
        
        elif 'file_input' in request.files and request.files['file_input'].filename:
            file = request.files['file_input']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                user_input = f.read()
            input_type = "file"
        
        elif 'pdf_input' in request.files and request.files['pdf_input'].filename:
            file = request.files['pdf_input']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            user_input = read_pdf(filepath)
            input_type = "pdf"
        
        elif 'image_input' in request.files and request.files['image_input'].filename:
            file = request.files['image_input']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            user_input = extract_text_from_image(filepath)
            input_type = "image"
        
        elif 'audio_input' in request.files and request.files['audio_input'].filename:
            file = request.files['audio_input']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            user_input = transcribe_audio(filepath)
            input_type = "audio"
        
        elif 'video_input' in request.files and request.files['video_input'].filename:
            file = request.files['video_input']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            user_input = summarize_video(filepath)
            input_type = "video"
        
        else:
            return render_template('index.html', error="Please provide some input")
        
        if not user_input.strip():
            return render_template('index.html', error="Empty input! Try again.")
        
        # Detect language and translate if needed
        try:
            language = detect(user_input)
            if language != "en":
                user_input = GoogleTranslator(source=language, target="en").translate(user_input)
        except:
            # If language detection fails, proceed with original text
            pass
        
        # Process and summarize text
        processed_text = preprocess_text(user_input)
        
        # Modified summarize_text function to accept max_words parameter instead of prompting
        def summarize_with_max_words(text, model_choice, max_words):
            from transformers import pipeline
            import torch
            from textwrap import wrap
            
            # Get the model name
            models = {
                "t5-small": "t5-small",
                "bart-large": "facebook/bart-large-cnn",
                "distilbart": "sshleifer/distilbart-cnn-12-6",
                "pegasus": "google/pegasus-xsum",
                "t5-3b": "t5-3b"
            }
            
            model_name = models.get(model_choice, "facebook/bart-large-cnn")
            model = pipeline("summarization", model=model_name, tokenizer=model_name, framework="pt", device=0 if torch.cuda.is_available() else -1)
            
            # Split text into manageable chunks and summarize each chunk
            text_chunks = wrap(text, width=500)
            summaries = []
            
            for chunk in text_chunks:
                try:
                    summary_output = model(chunk, max_length=max_words, min_length=10, do_sample=False)
                    summaries.append(summary_output[0]['summary_text'])
                except Exception as e:
                    summaries.append(chunk)  # Fallback to original chunk if summarization fails
            
            # Combine summaries and truncate to match the max word limit
            final_summary = " ".join(summaries)
            final_summary_words = final_summary.split()
            if len(final_summary_words) > max_words:
                final_summary = " ".join(final_summary_words[:max_words])
            
            return final_summary
        
        summary = summarize_with_max_words(processed_text, model_choice, max_words)
        
        # Perform additional analysis
        keywords = extract_keywords(user_input)[:10]  # Limit to top 10 keywords
        entities = extract_entities(user_input)[:15]  # Limit to top 15 entities
        sentiment = analyze_sentiment(user_input)
        topic = classify_topic(user_input)
        readability = readability_score(user_input)
        
        # Prepare result data
        result = {
            "summary": summary,
            "original_text": user_input,
            "analysis": {
                "keywords": keywords,
                "entities": entities,
                "sentiment": sentiment,
                "topic": topic,
                "readability": readability
            }
        }
        
        # Save results for later retrieval
        session_data[session_id] = {
            "result": result,
            "model_choice": model_choice,
            "input_type": input_type
        }
        
        # Save summary in different formats
        result_dir = os.path.join(RESULTS_FOLDER, session_id)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        save_as_txt(summary, os.path.join(result_dir, "summary.txt"))
        save_as_json(result, os.path.join(result_dir, "summary.json"))
        save_as_pdf(summary, os.path.join(result_dir, "summary.pdf"))
        save_as_docx(summary, os.path.join(result_dir, "summary.docx"))
        
        return render_template('result.html', result=result, session_id=session_id)
    
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/download/<session_id>/<format>')
def download(session_id, format):
    if session_id not in session_data:
        return redirect(url_for('index'))
    
    result_dir = os.path.join(RESULTS_FOLDER, session_id)
    
    if format == 'txt':
        return send_file(os.path.join(result_dir, "summary.txt"), as_attachment=True)
    elif format == 'json':
        return send_file(os.path.join(result_dir, "summary.json"), as_attachment=True)
    elif format == 'pdf':
        return send_file(os.path.join(result_dir, "summary.pdf"), as_attachment=True)
    elif format == 'docx':
        return send_file(os.path.join(result_dir, "summary.docx"), as_attachment=True)
    else:
        return redirect(url_for('index'))

@app.route('/ask_question/<session_id>', methods=['POST'])
def ask_question(session_id):
    if session_id not in session_data:
        return jsonify({"error": "Session expired or invalid"})
    
    question = request.form.get('question')
    if not question:
        return jsonify({"error": "No question provided"})
    
    original_text = session_data[session_id]['result']['original_text']
    answer = answer_question(original_text, question)
    
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)