import pytest
import os
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
os.environ["HF_TOKEN"] = "fake_hf_token"
os.environ["MOCK_LLM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import sentence_transformers
import numpy as np

class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.zeros(384)
        return np.zeros((len(texts), 384))

class MockCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, pairs, **kwargs):
        return np.zeros(len(pairs))

sentence_transformers.SentenceTransformer = MockSentenceTransformer
sentence_transformers.CrossEncoder = MockCrossEncoder

from app.main import app
from app.database.db import get_db, Base
from app.database.models import Document, Address, AddressDocument, DuplicateCandidate
from app.exceptions import LLMUnavailable
SQLALCHEMY_DATABASE_URL = "sqlite:///test_registry.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)

@pytest.fixture(name="db_session")

def db_session_fixture():
    try:
        if os.path.exists("test_registry.db"):
            os.remove("test_registry.db")

    except Exception:
        pass
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rejected_duplicates (
                sha256 TEXT PRIMARY KEY,
                rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS addresses_fts USING fts5(
                id UNINDEXED,
                raw_text
            );
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS address_after_insert AFTER INSERT ON addresses BEGIN
                INSERT INTO addresses_fts(id, raw_text) VALUES (new.id, new.raw_text);
            END;
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS address_after_delete AFTER DELETE ON addresses BEGIN
                DELETE FROM addresses_fts WHERE id = old.id;
            END;
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS address_after_update AFTER UPDATE OF raw_text ON addresses BEGIN
                UPDATE addresses_fts SET raw_text = new.raw_text WHERE id = old.id;
            END;
        """))
    db = TestingSessionLocal()
    try:
        yield db

    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS rejected_duplicates;"))
            conn.execute(text("DROP TABLE IF EXISTS addresses_fts;"))
        engine.dispose()
        try:
            if os.path.exists("test_registry.db"):
                os.remove("test_registry.db")

        except Exception:
            pass

@pytest.fixture(name="client")

def client_fixture(db_session):
    def override_get_db():
        try:
            yield db_session

        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app, raise_server_exceptions=True)
    app.dependency_overrides.clear()

def test_upload_success_and_duplicate_409(client):
    file_content = b"123 Main Street\nDallas, TX 75001"
    response = client.post(
        "/upload",
        files={"file": ("invoice_clean.txt", file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    doc_id = data["document_id"]
    response2 = client.post(
        "/upload",
        files={"file": ("invoice_clean_dup.txt", file_content, "text/plain")}
    )
    assert response2.status_code == 409
    assert "Duplicate file" in response2.json()["detail"]["message"]

def test_address_variants_deduplicate(client, db_session):
    response1 = client.post(
        "/upload",
        files={"file": ("doc1.txt", b"123 Main Street\nDallas, TX 75001", "text/plain")}
    )
    assert response1.status_code == 200
    response2 = client.post(
        "/upload",
        files={"file": ("doc2.txt", b"123 MAIN ST\n  DALLAS,   TX   75001", "text/plain")}
    )
    assert response2.status_code == 200
    addresses = db_session.query(Address).filter(Address.deleted_at == None).all()
    assert len(addresses) == 1
    assert addresses[0].normalized == "123 MAIN ST, DALLAS, TX 75001"
    links = db_session.query(AddressDocument).filter(AddressDocument.address_id == addresses[0].id).all()
    assert len(links) == 2

def test_failed_upload_stores_reason(client, db_session):
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    doc = db_session.query(Document).filter(Document.id == data["document_id"]).first()
    assert doc is not None
    assert doc.status == "failed"
    assert "empty" in doc.failure_reason.lower()

def test_merge_repoints_document_links(client, db_session):
    addr1 = Address(raw_text="123 Main St", normalized="123 MAIN ST", street="123 Main St", city="Dallas", state="TX", zip="75001")
    addr2 = Address(raw_text="123 Main Street", normalized="123 MAIN STREET", street="123 Main Street", city="Dallas", state="TX", zip="75001")
    db_session.add_all([addr1, addr2])
    db_session.commit()
    doc1 = Document(filename="doc1.txt", size_bytes=10, status="processed")
    doc2 = Document(filename="doc2.txt", size_bytes=12, status="processed")
    db_session.add_all([doc1, doc2])
    db_session.commit()
    link1 = AddressDocument(address_id=addr1.id, document_id=doc1.id)
    link2 = AddressDocument(address_id=addr2.id, document_id=doc2.id)
    db_session.add_all([link1, link2])
    db_session.commit()
    candidate = DuplicateCandidate(address1_id=addr1.id, address2_id=addr2.id, score=90, status="pending")
    db_session.add(candidate)
    db_session.commit()
    response = client.post(
        f"/duplicates/{candidate.id}/resolve",
        json={"action": "merge"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Addresses merged"
    db_session.refresh(candidate)
    assert candidate.status == "merged"
    links_updated = db_session.query(AddressDocument).filter(AddressDocument.document_id == doc2.id).all()
    assert len(links_updated) == 1
    assert links_updated[0].address_id == addr1.id

def test_mock_llm_extract_endpoint(client, db_session, monkeypatch):

    def mock_generate_extractor(messages, max_tokens=300):
        return '{"addresses": [{"street": "999 Oak Rd", "city": "Columbus", "state": "OH", "zip": "43210"}]}'
    monkeypatch.setattr("app.extractor.generate", mock_generate_extractor)
    doc = Document(
        filename="test_llm.txt",
        size_bytes=50,
        status="processed",
        content="Burying address 999 Oak Road, Columbus, Ohio, 43210 here."
    )
    db_session.add(doc)
    db_session.commit()
    response = client.post(f"/documents/{doc.id}/extract_llm")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LLM_SUCCESS"
    assert data["addresses"]["addresses"][0]["street"] == "999 OAK RD"

def test_mock_rag_ask_endpoint(client, db_session, monkeypatch):

    def mock_generate_qa(messages, max_tokens=150):
        return '{"answer": "1600 Pennsylvania Ave", "sources": ["letter_dc.txt"], "context_found": true}'

    def mock_retrieve(question, top_k=4):
        return [{"filename": "letter_dc.txt", "chunk": "The address is 1600 Pennsylvania Ave.", "score": 0.99}]
    monkeypatch.setattr("app.rag.retrieve", mock_retrieve)
    monkeypatch.setattr("app.rag.generate", mock_generate_qa)
    response = client.post("/ask", json={"question": "What is the forwarding address?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "1600 Pennsylvania Ave"
    assert data["sources"] == ["letter_dc.txt"]
    assert data["context_found"] is True

