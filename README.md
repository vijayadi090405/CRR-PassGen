# CRR-PassGen #                                        
                                                                                                                       

  CRR PassGen - Human-Memorable Password Generator
  (Built with Whisper, GPT-2, SpaCy, Flask, and Python)

--------------------------------------------------

 Project Title: CRR PassGen
 Author: Vijay Aditya P.
 Description: Generate secure, smart, and memorable passwords 
               from user information found in audio, video, or 
               document files.

--------------------------------------------------

 SETUP INSTRUCTIONS

1️⃣ Download the zip file and extract it.

2️⃣ Install Python libraries:
    > pip install -r requirements.txt

3️⃣ Run the Flask application:
    > python app.py

4️⃣ Open browser and go to:
    http://127.0.0.1:5000

--------------------------------------------------

 FEATURES

 Upload audio, video, or document (.mp4, .mp3, .pdf, .docx)
 Automatically clears the uploads folder every 30 minutes to save disk space and ensure user privacy.
 Transcribe speech using OpenAI Whisper
 Extract people/places/orgs/dates using SpaCy and GPT-2
 Suggest upto 10 strong, personalized & memorable passwords each time
 Analyze each password’s brute-force resistance

--------------------------------------------------

 TECHNOLOGIES USED

- Python, Flask
- OpenAI Whisper (Tiny)
- SpaCy (en_core_web_sm)
- GPT-2 via Hugging Face Transformers
- pdfplumber, python-docx, torch, pydub

--------------------------------------------------

 DISCLAIMER

This tool is for research and educational use only.
Do not use for unauthorized access or unethical purposes.

--------------------------------------------------

🎵 Stay secure, stay creative!
