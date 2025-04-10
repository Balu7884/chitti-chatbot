import gradio as gr
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader("Data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = ''' You are a Strategic thinking project manager. Respond through
    {context}
    user : {question}
    ChatBot :
    '''
    prompt = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

print("Initializing Chatbot")
llm = initialize_llm()
db_path = './chroma_db'

if not os.path.exists(db_path):
    print("Creating ChromaDB")
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

def chatbot(user_input, history=None):
    if history is None:
        history = []

    if not user_input or not user_input.strip():
        response = "Hey! I am Babji your assistant. Please ask me something."
    else:
        response = qa_chain.run(user_input)

    if isinstance(response, tuple):
        response = response[0]

    if not isinstance(response, str):
        response = "Sorry, I couldn't understand your request."

    history.append({"role": "assistant", "content": response})
    return response

with gr.Blocks() as demo:
    gr.Markdown("# Babji - Your Project Assistant")
    with gr.Row():
        input_text = gr.Textbox(label="Type your message")
    submit_button = gr.Button("Submit")
    output_text = gr.Textbox(label="Babji's Response")
    submit_button.click(chatbot, inputs=[input_text], outputs=[output_text])

demo.launch()
