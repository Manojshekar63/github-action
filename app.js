import streamlit as st
import tempfile
import os
import threading

from langchain_community.document_loaders import UnstructuredPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# Voice input support
from streamlit_mic_recorder import speech_to_text

# Text-to-Speech (offline)
import pyttsx3

# Set Poppler path for Windows PDF processing
POPPLER_PATH = r"C:\poppler-25.07.0\Library\bin"
os.environ["POPPLER_PATH"] = POPPLER_PATH

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Global flag to control speech
stop_speaking = False

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = UnstructuredPDFLoader(
        tmp_file_path,
        unstructured_kwargs={"pdf_extractor": "pdf2image"}
    )
    docs = loader.load()
    os.unlink(tmp_file_path)
    return docs

def extract_text_from_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = Docx2txtLoader(tmp_file_path)
    docs = loader.load()
    os.unlink(tmp_file_path)
    return docs

def speak_text(text):
    """Speak text using pyttsx3 in a separate thread with stop support"""
    global stop_speaking
    tts_engine.connect('started-utterance', lambda name: None if not stop_speaking else False)
    tts_engine.say(text, 'speech')
    tts_engine.startLoop()

    # Run loop until speech ends or stop is triggered
    while not stop_speaking:
        try:
            tts_engine.iterate()
        except:
            break

    tts_engine.endLoop()

def stop_speech():
    """Set flag to stop speech"""
    global stop_speaking
    stop_speaking = True

def reset_speech():
    """Reset stop flag for next speech"""
    global stop_speaking
    stop_speaking = False

def main():
    st.title("LangChain PDF Q&A Chatbot with Llama 3.1 (Ollama)")

    # Choose interaction language
    lang_choice = st.selectbox("Interaction language", ["English", "Kannada"], index=0)

    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])

    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == "pdf":
            with st.spinner("Extracting text from PDF..."):
                docs = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            with st.spinner("Extracting text from Word document..."):
                docs = extract_text_from_docx(uploaded_file)

        st.write("### Document preview:")
        preview_text = "\n\n".join(doc.page_content for doc in docs[:2])
        st.text_area("Preview of first two pages (truncated):", preview_text[:1000] + "...", height=200)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)

        # Create embeddings and vector store (multilingual model)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectordb = FAISS.from_documents(texts, embeddings)

        # Initialize Ollama Llama 3.1 model
        llm = ChatOllama(model="llama3.1", temperature=0)

        # Setup retrieval-based QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        # Voice input with mic icon
        col1, col2 = st.columns([1, 8])
        with col1:
            st.markdown('<span title="Record your voice question"><span style="font-size:35px; cursor:pointer;">🎤</span></span>', unsafe_allow_html=True)
        with col2:
            st.write("Say your question or type below:")

        # Capture voice input (adapt language code)
        if lang_choice == "Kannada":
            voice_query = speech_to_text(language='kn', just_once=False, key='voice_input')
        else:
            voice_query = speech_to_text(language='en', just_once=False, key='voice_input')

        # Show voice transcription or allow typing
        if voice_query:
            query = st.text_input("Recognized voice input:", value=voice_query, key="voice_box")
        else:
            query = st.text_input("Type your question:", key="type_box")

        final_query = query.strip()

        if final_query:
            # If Kannada, translate question to English for retrieval if documents are English
            retrieval_query = final_query
            if lang_choice == "Kannada":
                trans_q = llm.invoke(
                    "Translate the following Kannada question to English. Reply ONLY with the English translation:\n\n"
                    + final_query
                )
                retrieval_query = getattr(trans_q, "content", str(trans_q))

            with st.spinner("Generating answer..."):
                result = qa_chain.invoke({"query": retrieval_query})
                if isinstance(result, dict):
                    answer = result.get("result") or result.get("answer") or ""
                else:
                    answer = str(result)

            # If Kannada requested, translate answer back
            if lang_choice == "Kannada":
                trans_a = llm.invoke(
                    "Translate the following answer to Kannada. Reply ONLY with the Kannada translation:\n\n"
                    + answer
                )
                answer = getattr(trans_a, "content", str(trans_a))

            st.write("*Answer:*", answer)

            # Buttons for TTS control
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔊 Hear Answer"):
                    reset_speech()
                    # Run TTS in a separate thread
                    thread = threading.Thread(target=speak_text, args=(answer,), daemon=True)
                    thread.start()

            with col2:
                if st.button("🛑 Stop Hearing"):
                    stop_speech()

if __name__ == "__main__":
    main()