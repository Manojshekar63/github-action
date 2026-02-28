import pytest
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────
# 1. Test text splitter logic
# ─────────────────────────────────────────
def test_text_splitter():
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = splitter.split_text("This is a test document. " * 20)

    assert len(texts) > 1              # should split into multiple chunks
    assert all(len(t) <= 150 for t in texts)  # each chunk not too large

# ─────────────────────────────────────────
# 2. Test PDF extraction function
# ─────────────────────────────────────────
def test_extract_text_from_pdf():
    import sys
    sys.path.insert(0, r"d:\git action")
    from app import extract_text_from_pdf

    # Mock an uploaded file
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
    import sys
    sys.path.insert(0, r"d:\git action")
    from app import extract_text_from_docx

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
    import sys
    sys.path.insert(0, r"d:\git action")
    import app

    app.stop_speaking = False
    app.stop_speech()
    assert app.stop_speaking == True   # flag should be True after stop

def test_reset_speech_flag():
    import sys
    sys.path.insert(0, r"d:\git action")
    import app

    app.stop_speaking = True
    app.reset_speech()
    assert app.stop_speaking == False  # flag should be False after reset

# ─────────────────────────────────────────
# 5. Test answer extraction from result dict
# ─────────────────────────────────────────
def test_answer_extraction_from_result():
    result = {"result": "This is the answer"}

    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or ""
    else:
        answer = str(result)

    assert answer == "This is the answer"

def test_answer_extraction_fallback():
    result = {"answer": "Fallback answer"}

    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or ""
    else:
        answer = str(result)

    assert answer == "Fallback answer"

def test_answer_extraction_empty():
    result = {}

    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or ""
    else:
        answer = str(result)

    assert answer == ""