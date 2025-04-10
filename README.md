---
title: Chitti
emoji: 👀
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# 🤖 Chitti - AI Therapist Chatbot

Welcome to **Chitti**, your personal AI-powered therapist chatbot built with 💬 LangChain, 🧠 LLMs, and an intuitive interface. Designed to simulate therapeutic conversations in a supportive and safe environment.

## 🧠 About the Project

This project leverages:
- **LangChain** for natural language processing and chaining.
- **Gradio** for building a simple and elegant web interface.
- **ChromaDB** for storing and retrieving contextual data.
- **OpenAI** or any supported LLMs for chat responses.

Chitti is not a replacement for professional therapy but can be a helpful companion for conversation and reflection.

---

## 🚀 Features

- Real-time AI chatbot for mental health conversations
- Context-aware dialogue flow
- Easy-to-use web UI built with Gradio
- Local vector database storage using ChromaDB
- Lightweight and extendable

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/Balu7884/chitti-chatbot.git
cd chitti-chatbot
Create a virtual environment and activate it:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate   # On Mac/Linux
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🧪 Run the App
To launch the chatbot locally:

bash
Copy
Edit
python app.py
Or use:

bash
Copy
Edit
gradio app.py
📁 Folder Structure
bash
Copy
Edit
├── app.py                   # Main app file
├── chroma_db/               # Vector DB for storing chat context
├── requirements.txt         # Python dependencies
├── README.md                # You're reading it!
└── ...
🛑 Disclaimer
This chatbot is for educational purposes only. It does not provide medical advice or replace professional mental health services.

📜 License
MIT License

🙌 Acknowledgements
LangChain

Chroma

Gradio

OpenAI
