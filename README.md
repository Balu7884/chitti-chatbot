# Babji - Your Project Assistant

Babji is an AI-powered project assistant designed to help project managers and teams efficiently manage their projects. Leveraging advanced language models and document retrieval, Babji can answer questions based on project documentation, providing strategic insights and support throughout the project lifecycle.

This repository contains the full stack of the Babji application, including a Python Flask backend for AI logic and a React frontend for an intuitive user interface.

## Features

* **Intelligent Q&A**: Get instant answers to project-related questions by querying your project documentation (PDFs).
* [cite_start]**Project Management Knowledge**: Built upon comprehensive project management principles outlined in "Project Management - 2nd Edition"[cite: 1].
* **Scalable Architecture**: Separated backend (Python Flask) and frontend (React) for easy development, deployment, and scaling.
* **Local Vector Database**: Uses ChromaDB for efficient storage and retrieval of document embeddings, enabling fast and relevant responses.
* **Conversational Interface**: A user-friendly chat interface built with React.

## Technologies Used

### Backend
* **Python**: Core programming language.
* **Flask**: Web framework for building the API.
* **LangChain**: Framework for developing applications powered by language models.
* **Groq API**: For fast and efficient inference with large language models (LLMs).
* **HuggingFace BGE Embeddings**: For generating document embeddings.
* **ChromaDB**: Vector database for storing and querying document embeddings.
* **PyPDFLoader**: For loading PDF documents.
* **python-dotenv**: For managing environment variables.

### Frontend
* **React**: JavaScript library for building user interfaces.
* **HTML/CSS**: For structuring and styling the web application.
* **`fetch` API**: For communicating with the Flask backend.

## Project Structure

y-project-assistant/
├── Data/
│   └── Project-Management-2nd-Edition-1729807212.pdf
├── AiTherapist/                  ← Python backend (API, logic, etc.)
│   ├── app.py
│   ├── requirements.txt
│   ├── .env.example
│   └── chroma_db/                ← Likely vector DB (e.g., for embeddings), ignored by Git
├── my-project-assistant-frontend/ ← React frontend
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   ├── .env.example
│   ├── package.json
│   └── node_modules/             ← Ignored by Git
├── .gitignore
└── README.md



## Setup and Installation

Follow these steps to get Babji up and running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/Balu7884/AiTherapist.git](https://github.com/Balu7884/AiTherapist.git)
cd AiTherapist
2. Backend Setup (Python)
Navigate to the backend directory:

Bash

cd AiTherapist/AiTherapist # Adjust if your main project folder is different
Create a Python Virtual Environment:

Bash

python -m venv venv
Activate the Virtual Environment:

Windows:

Bash

.\venv\Scripts\activate
macOS/Linux:

Bash

source venv/bin/activate
Install Python Dependencies:

Bash

pip install -r requirements.txt
Configure Environment Variables:

Create a file named .env in the AiTherapist/AiTherapist directory (same level as app.py).

Add your Groq API Key to this file:

GROQ_API_KEY="gsk_YOUR_ACTUAL_GROQ_API_KEY_HERE"
Replace "gsk_YOUR_ACTUAL_GROQ_API_KEY_HERE" with your actual Groq API key. Do not commit your actual .env file to Git!

Place your PDF Documentation:

Ensure your Project-Management-2nd-Edition-1729807212.pdf file is located in the Data/ directory. If you have other relevant PDFs, place them here too. The create_vector_db() function will load all .pdf files from this directory.

Run the Flask Backend Server:

Bash

python app.py
The backend server will start, typically on http://127.0.0.1:5000/. You should see "ChromaDB created and data saved" or "ChromaDB loaded successfully" messages in your terminal.

3. Frontend Setup (React)
Open a new terminal window (keep the backend running in the first one).

Navigate to the frontend directory:

Bash

cd AiTherapist/my-project-assistant-frontend # Adjust path if different
Install Node.js Dependencies:

Bash

npm install
Run the React Development Server:

Bash

npm start
The React app will open in your web browser, usually at http://localhost:3000/.

Usage
Ensure both the Python backend and React frontend servers are running.

Open your web browser to http://localhost:3000/.

Type your project-related questions into the chat input field and press Enter or click "Send".

Babji will process your query using the project documentation and provide a response.

Important Notes & Troubleshooting
API Key Security: Your GROQ_API_KEY is sensitive. Never hardcode it directly in your source files or commit your .env file to Git. The .gitignore file is configured to prevent this. If you accidentally commit it, you will need to revoke the key on Groq's platform and rewrite your Git history.

Virtual Environment: The venv/ folder is ignored by Git. Do not manually add it.

ChromaDB Persistence: The chroma_db/ directory stores your vector database and is ignored by Git. If you clone the repository or delete chroma_db/, the backend will automatically recreate it by processing your PDFs when app.py runs.

Kernel Crashes: If your Python kernel crashes, especially after installing packages, try these steps:

Ensure all required Python packages are installed from requirements.txt.

Check for conflicts (e.g., PyPDF2 vs. pypdf). Uninstall both and reinstall only pypdf via requirements.txt after updating the requirements.txt file.

Delete and recreate your Python virtual environment (venv/), then reinstall dependencies.

Ensure load_dotenv() is called before os.getenv() in your scripts.

CORS Issues: If your frontend cannot communicate with the backend, ensure flask_cors is installed (pip install flask-cors) and CORS(app) is enabled in app.py.

Contributing
Contributions are welcome! If you find bugs, have feature requests, or want to improve the code, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
