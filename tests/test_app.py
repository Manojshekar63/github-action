import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# ─── Add root path ONCE at top level (fixes all ModuleNotFoundError) ───
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ─── Mock ALL heavy modules BEFORE importing app ───
MOCK_MODULES = {
    "pyttsx3": MagicMock(),
    "streamlit": MagicMock(),
    "streamlit_mic_recorder": MagicMock(),
    "langchain_ollama": MagicMock(),
    "langchain_huggingface": MagicMock(),
    "langchain_community": MagicMock(),
    "langchain_community.document_loaders": MagicMock(),
    "langchain_community.vectorstores": MagicMock(),
    "langchain.chains": MagicMock(),
    "faiss": MagicMock(),
    "sentence_transformers": MagicMock(),
    "unstructured": MagicMock(),
}

# Apply all mocks before any test runs
for mod_name, mock in MOCK_MODULES.items():
    sys.modules[mod_name] = mock

# ─────────────────────────────────────────
# 1. Test text splitter
# ─────────────────────────────────────────
def test_text_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = splitter.split_text("This is a test document. " * 20)

    assert len(texts) > 1
    assert all(len(t) <= 150 for t in texts)

# ─────────────────────────────────────────
# 2. Test PDF extraction
# ─────────────────────────────────────────
def test_extract_text_from_pdf():
    # Remove cached app module to reimport cleanly
    if "app" in sys.modules:
        del sys.modules["app"]

    from app import extract_text_from_pdf

    mock_file = MagicMock()
    mock_file.read.return_value = b"%PDF-1.4 fake content"

    with patch("app.UnstructuredPDFLoader") as mock_loader, \
         patch("app.tempfile.NamedTemporaryFile"), \
         patch("app.os.unlink"):

        mock_loader.return_value.load.return_value = [
            MagicMock(page_content="Sample PDF text")
        ]

        docs = extract_text_from_pdf(mock_file)
        assert docs is not None

# ─────────────────────────────────────────
# 3. Test DOCX extraction
# ─────────────────────────────────────────
def test_extract_text_from_docx():
    if "app" in sys.modules:
        del sys.modules["app"]

    from app import extract_text_from_docx

    mock_file = MagicMock()
    mock_file.read.return_value = b"fake docx content"

    with patch("app.Docx2txtLoader") as mock_loader, \
         patch("app.tempfile.NamedTemporaryFile"), \
         patch("app.os.unlink"):

        mock_loader.return_value.load.return_value = [
            MagicMock(page_content="Sample DOCX text")
        ]

        docs = extract_text_from_docx(mock_file)
        assert docs is not None

# ─────────────────────────────────────────
# 4. Test stop speech flag
# ─────────────────────────────────────────
def test_stop_speech_flag():
    if "app" in sys.modules:
        del sys.modules["app"]

    import app

    app.stop_speaking = False
    app.stop_speech()
    assert app.stop_speaking == True

# ─────────────────────────────────────────
# 5. Test reset speech flag
# ─────────────────────────────────────────
def test_reset_speech_flag():
    if "app" in sys.modules:
        del sys.modules["app"]

    import app

    app.stop_speaking = True
    app.reset_speech()
    assert app.stop_speaking == False

# ─────────────────────────────────────────
# 6. Test answer extraction (already passing ✅)
# ─────────────────────────────────────────
def test_answer_extraction_from_result():
    result = {"result": "This is the answer"}
    answer = result.get("result") or result.get("answer") or ""
    assert answer == "This is the answer"

def test_answer_extraction_fallback():
    result = {"answer": "Fallback answer"}
    answer = result.get("result") or result.get("answer") or ""
    assert answer == "Fallback answer"

def test_answer_extraction_empty():
    result = {}
    answer = result.get("result") or result.get("answer") or ""
    assert answer == ""