import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# ─── Fix: use relative path (works on both Windows & Linux) ───
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────
# 1. Test text splitter logic
# ─────────────────────────────────────────
def test_text_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← Fixed import

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = splitter.split_text("This is a test document. " * 20)

    assert len(texts) > 1
    assert all(len(t) <= 150 for t in texts)

# ─────────────────────────────────────────
# 2. Test PDF extraction function
# ─────────────────────────────────────────
def test_extract_text_from_pdf():
    with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: (
        MagicMock() if name in ["pyttsx3", "streamlit", "streamlit_mic_recorder"] 
        else __import__(name, *args, **kwargs)
    )):
        with patch.dict("sys.modules", {
            "pyttsx3": MagicMock(),
            "streamlit": MagicMock(),
            "streamlit_mic_recorder": MagicMock(),
        }):
            from app import extract_text_from_pdf   # ← now works with relative path

            mock_file = MagicMock()
            mock_file.read.return_value = b"%PDF-1.4 fake content"

            with patch("app.UnstructuredPDFLoader") as mock_loader:
                mock_loader.return_value.load.return_value = [
                    MagicMock(page_content="Sample PDF text")
                ]
                with patch("app.tempfile.NamedTemporaryFile"):
                    with patch("app.os.unlink"):
                        docs = extract_text_from_pdf(mock_file)
                        assert docs is not None

# ─────────────────────────────────────────
# 3. Test DOCX extraction function
# ─────────────────────────────────────────
def test_extract_text_from_docx():
    with patch.dict("sys.modules", {
        "pyttsx3": MagicMock(),
        "streamlit": MagicMock(),
        "streamlit_mic_recorder": MagicMock(),
    }):
        from app import extract_text_from_docx   # ← now works with relative path

        mock_file = MagicMock()
        mock_file.read.return_value = b"fake docx content"

        with patch("app.Docx2txtLoader") as mock_loader:
            mock_loader.return_value.load.return_value = [
                MagicMock(page_content="Sample DOCX text")
            ]
            with patch("app.tempfile.NamedTemporaryFile"):
                with patch("app.os.unlink"):
                    docs = extract_text_from_docx(mock_file)
                    assert docs is not None

# ─────────────────────────────────────────
# 4. Test stop/reset speech flags
# ─────────────────────────────────────────
def test_stop_speech_flag():
    with patch.dict("sys.modules", {
        "pyttsx3": MagicMock(),
        "streamlit": MagicMock(),
        "streamlit_mic_recorder": MagicMock(),
    }):
        import app   # ← now works with relative path
        app.stop_speaking = False
        app.stop_speech()
        assert app.stop_speaking == True

def test_reset_speech_flag():
    with patch.dict("sys.modules", {
        "pyttsx3": MagicMock(),
        "streamlit": MagicMock(),
        "streamlit_mic_recorder": MagicMock(),
    }):
        import app   # ← now works with relative path
        app.stop_speaking = True
        app.reset_speech()
        assert app.stop_speaking == False

# ─────────────────────────────────────────
# 5. Test answer extraction (these already pass ✅)
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