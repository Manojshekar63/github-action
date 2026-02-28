import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# ─── Add root path ───
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ─── Mock heavy modules BEFORE importing helpers ───
MOCK_MODULES = {
    "langchain_community": MagicMock(),
    "langchain_community.document_loaders": MagicMock(),
    "langchain_community.vectorstores": MagicMock(),
}

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
    if "helpers" in sys.modules:
        del sys.modules["helpers"]

    from helpers import extract_text_from_pdf

    mock_file = MagicMock()
    mock_file.read.return_value = b"%PDF-1.4 fake content"

    with patch("helpers.UnstructuredPDFLoader") as mock_loader, \
         patch("helpers.tempfile.NamedTemporaryFile"), \
         patch("helpers.os.unlink"):

        mock_loader.return_value.load.return_value = [
            MagicMock(page_content="Sample PDF text")
        ]

        docs = extract_text_from_pdf(mock_file)
        assert docs is not None
        assert len(docs) == 1

# ─────────────────────────────────────────
# 3. Test DOCX extraction
# ─────────────────────────────────────────
def test_extract_text_from_docx():
    if "helpers" in sys.modules:
        del sys.modules["helpers"]

    from helpers import extract_text_from_docx

    mock_file = MagicMock()
    mock_file.read.return_value = b"fake docx content"

    with patch("helpers.Docx2txtLoader") as mock_loader, \
         patch("helpers.tempfile.NamedTemporaryFile"), \
         patch("helpers.os.unlink"):

        mock_loader.return_value.load.return_value = [
            MagicMock(page_content="Sample DOCX text")
        ]

        docs = extract_text_from_docx(mock_file)
        assert docs is not None
        assert len(docs) == 1

# ─────────────────────────────────────────
# 4. Test stop speech flag
# ─────────────────────────────────────────
def test_stop_speech_flag():
    if "helpers" in sys.modules:
        del sys.modules["helpers"]

    import helpers

    helpers.stop_speaking = False
    helpers.stop_speech()
    assert helpers.stop_speaking == True

# ─────────────────────────────────────────
# 5. Test reset speech flag
# ─────────────────────────────────────────
def test_reset_speech_flag():
    if "helpers" in sys.modules:
        del sys.modules["helpers"]

    import helpers

    helpers.stop_speaking = True
    helpers.reset_speech()
    assert helpers.stop_speaking == False

# ─────────────────────────────────────────
# 6. Test answer extraction
# ─────────────────────────────────────────
def test_answer_extraction_from_result():
    if "helpers" in sys.modules:
        del sys.modules["helpers"]

    from helpers import extract_answer

    assert extract_answer({"result": "This is the answer"}) == "This is the answer"

def test_answer_extraction_fallback():
    from helpers import extract_answer

    assert extract_answer({"answer": "Fallback answer"}) == "Fallback answer"

def test_answer_extraction_empty():
    from helpers import extract_answer

    assert extract_answer({}) == ""

def test_answer_extraction_string():
    from helpers import extract_answer

    assert extract_answer("plain string") == "plain string"