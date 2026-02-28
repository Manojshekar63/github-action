import tempfile
import os
from langchain_community.document_loaders import UnstructuredPDFLoader, Docx2txtLoader

# Global flag to control speech
stop_speaking = False

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = UnstructuredPDFLoader(tmp_file_path)
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

def stop_speech():
    global stop_speaking
    stop_speaking = True

def reset_speech():
    global stop_speaking
    stop_speaking = False

def extract_answer(result):
    if isinstance(result, dict):
        return result.get("result") or result.get("answer") or ""
    return str(result)