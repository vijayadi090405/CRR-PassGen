
import os
import re
import whisper
import spacy
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import pdfplumber
import docx
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import string
import torch
import secrets

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Tiny Whisper
model = whisper.load_model("tiny")

# Load SpaCy NER
nlp = spacy.load("en_core_web_sm")

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt_model.config.pad_token_id = tokenizer.eos_token_id

def transcribe_audio(path):
    result = model.transcribe(path)
    return result["text"]

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_entities_spacy(text):
    doc = nlp(text)
    info = {
        "names": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
        "locations": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
        "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        "nouns": list(set([token.text for token in doc if token.pos_ == "NOUN"]))
    }
    return info

def extract_entities_llm(text):
    prompt = (
        "Extract the following entities from the text delimited by triple backticks. "
        "Return exactly four lists labeled Names, Dates, Locations, and Organizations, "
        "each as a comma-separated list. If none, leave empty.\n\n"
        "Text: ```" + text + "```\n\n"
        "Names:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            do_sample=False,
            num_return_sequences=1
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    entities = {"names": [], "dates": [], "locations": [], "organizations": []}
    for key in entities.keys():
        pattern = re.compile(key.capitalize() + r":([^\n]*)", re.IGNORECASE)
        match = pattern.search(decoded)
        if match:
            items = match.group(1).strip()
            if items:
                entities[key] = [item.strip() for item in items.split(",") if item.strip()]
    return entities

def merge_entities(spacy_entities, llm_entities):
    merged = {}
    for key in spacy_entities.keys():
        combined = set(e.strip() for e in spacy_entities.get(key, []) if e.strip())
        combined.update(e.strip() for e in llm_entities.get(key, []) if e.strip())
        merged[key] = list(combined)
    return merged

def generate_passwords_from_details(details, count=10):
    # Split the details into words and extract relevant parts
    words = re.findall(r'\w+', details)  # Extract words from the details
    if not words:
        return []  # Return empty if no words found

    # Create a list to hold generated passwords
    passwords = set()

    # Generate passwords using combinations of the extracted words
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j:
                # Combine two different words with a number or special character
                passwords.add(f"{words[i]}{words[j]}{secrets.randbelow(100)}")  # Add a random number
                passwords.add(f"{words[i]}_{words[j]}")  # Underscore combination
                passwords.add(f"{words[i]}{secrets.choice(['@', '#', '$', '%'])}{words[j]}")  # Special character

    # Convert the set to a list and filter based on length and criteria
    passwords = list(passwords)
    passwords = [
        pwd for pwd in passwords
        if 8 <= len(pwd) <= 20 and re.search(r"\d", pwd) and re.search(r"[A-Za-z]", pwd) and " " not in pwd
    ][:count]

    return passwords
# Password analysis functions unchanged...

charset = string.printable[:-6]

def brute_force_position(password, charset):
    position = 0
    found = False
    max_length = len(password)
    if max_length > 4:
        base = len(charset)
        pos = 0
        for length in range(1, max_length):
            pos += base ** length
        index = 0
        for ch in password:
            index *= base
            try:
                index += charset.index(ch)
            except ValueError:
                index += 0
        pos += index + 1
        return pos
    else:
        for length in range(1, max_length + 1):
            for attempt in product(charset, repeat=length):
                position += 1
                if ''.join(attempt) == password:
                    found = True
                    break
            if found:
                break
        return position
def estimate_time(password, guesses_per_second=1_000_000):
    pos = brute_force_position(password, charset)
    seconds = pos / guesses_per_second
    return pos, seconds

def human_time(seconds):
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days > 0:
        return f"{int(days)}d {int(hrs)}h {int(mins)}m {int(secs)}s"
    elif hrs > 0:
        return f"{int(hrs)}h {int(mins)}m {int(secs)}s"
    elif mins > 0:
        return f"{int(mins)}m {int(secs)}s"
    else:
        return f"{int(secs)}s"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filetype = request.form['type']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    if filetype == "video":
        audio = AudioSegment.from_file(path)
        audio_path = path.rsplit('.', 1)[0] + ".wav"
        audio.export(audio_path, format="wav")
        text = transcribe_audio(audio_path)
    elif filetype == "audio":
        text = transcribe_audio(path)
    elif filetype == "document":
        if path.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif path.endswith(".docx") or path.endswith(".doc"):
            text = extract_text_from_docx(path)
        else:
            return jsonify({"error": "Unsupported file format."}), 400
    else:
        return jsonify({"error": "Invalid type."}), 400

    return jsonify({"text": text})

@app.route('/extract', methods=['POST'])
def extract():
    text = request.json.get("text", "")
    spacy_results = extract_entities_spacy(text)
    llm_results = extract_entities_llm(text)
    merged = merge_entities(spacy_results, llm_results)
    return jsonify(merged)

@app.route('/generate-passwords', methods=['POST'])
def generate():
    details = request.json.get("text", "")
    passwords = generate_passwords_from_details(details, 10)
    return jsonify(passwords)

@app.route('/analyze-password', methods=['POST'])
def analyze_password():
    password = request.json.get("password", "")
    pos, time_required = estimate_time(password)
    return jsonify({
        "position": pos,
        "time_required": time_required,
        "human_time": human_time(time_required)
    })

import threading
import time

def clear_uploads_periodically(folder=UPLOAD_FOLDER, interval_minutes=30):
    def clear_folder():
        while True:
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now}] Clearing upload folder...")
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
            time.sleep(interval_minutes * 60)
    thread = threading.Thread(target=clear_folder, daemon=True)
    thread.start()


if __name__ == "__main__":
    app.run(debug=True)
