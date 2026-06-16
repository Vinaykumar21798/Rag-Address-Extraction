from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.database.models import (
    Base,
    Document,
    Address,
    AddressDocument,
    DuplicateCandidate,
    RagLog,
)
DATABASE_URL = "sqlite:///registry.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def init_db():
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
        conn.execute(text("""
            INSERT INTO addresses_fts(id, raw_text)
            SELECT id, raw_text FROM addresses
            WHERE id NOT IN (SELECT id FROM addresses_fts);
        """))
        try:
            cursor = conn.execute(text("PRAGMA table_info(duplicate_candidates);"))
            columns = [row[1] for row in cursor.fetchall()]
            if "recommendation" not in columns:
                conn.execute(text("ALTER TABLE duplicate_candidates ADD COLUMN recommendation TEXT DEFAULT 'manual_review';"))

        except Exception:
            pass
        try:
            cursor = conn.execute(text("PRAGMA table_info(documents);"))
            columns = [row[1] for row in cursor.fetchall()]
            if "content" not in columns:
                conn.execute(text("ALTER TABLE documents ADD COLUMN content TEXT;"))

        except Exception:
            pass

def get_db():
    db = SessionLocal()
    try:
        yield db

    finally:
        db.close()

def create_document(
    db,
    filename,
    size_bytes,
    sha256=None,
    content_hash=None,
    status="processed",
    failure_reason=None,
    content=None
):
    document = Document(
        filename=filename,
        size_bytes=size_bytes,
        sha256=sha256,
        content_hash=content_hash,
        status=status,
        failure_reason=failure_reason,
        content=content
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document

def create_address(
    db,
    raw_text,
    normalized,
    street,
    city,
    state,
    zip_code
):
    address = Address(
        raw_text=raw_text,
        normalized=normalized,
        street=street,
        city=city,
        state=state,
        zip=zip_code
    )
    db.add(address)
    db.commit()
    db.refresh(address)
    return address

def create_address_document_link(
    db,
    address_id,
    document_id
):
    existing = (
        db.query(AddressDocument)
        .filter(
            AddressDocument.address_id == address_id,
            AddressDocument.document_id == document_id
        )
        .first()
    )
    if existing:
        return existing
    link = AddressDocument(
        address_id=address_id,
        document_id=document_id
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link

def get_address_by_normalized(
    db,
    normalized
):
    return (
        db.query(Address)
        .filter(
            Address.normalized == normalized
        )
        .first()
    )

def get_documents(db):
    return db.query(Document).all()

def get_document_by_id(db, document_id):
    return (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )

def get_document_by_sha256(db, sha256):
    return (
        db.query(Document)
        .filter(Document.sha256 == sha256)
        .first()
    )

def get_addresses(
    db,
    limit=20,
    offset=0,
    search=None,
    city=None,
    state=None,
    zip_code=None
):
    query = (
        db.query(Address)
        .filter(Address.deleted_at == None)
    )
    if search:
        words = [w for w in search.strip().split() if w]
        if words:
            escaped_words = [w.replace('"', '""') for w in words]
            fts_query = " AND ".join(f'"{w}"*' for w in escaped_words)
            query = query.filter(
                text("addresses.id IN (SELECT id FROM addresses_fts WHERE addresses_fts MATCH :fts_query)")
            ).params(fts_query=fts_query)
    if city:
        query = query.filter(
            Address.city.ilike(city)
        )
    if state:
        query = query.filter(
            Address.state.ilike(state)
        )
    if zip_code:
        query = query.filter(
            Address.zip == zip_code
        )
    return (
        query
        .offset(offset)
        .limit(limit)
        .all()
    )

def get_addresses_count(
    db,
    search=None,
    city=None,
    state=None,
    zip_code=None
):
    query = (
        db.query(Address)
        .filter(Address.deleted_at == None)
    )
    if search:
        words = [w for w in search.strip().split() if w]
        if words:
            escaped_words = [w.replace('"', '""') for w in words]
            fts_query = " AND ".join(f'"{w}"*' for w in escaped_words)
            query = query.filter(
                text("addresses.id IN (SELECT id FROM addresses_fts WHERE addresses_fts MATCH :fts_query)")
            ).params(fts_query=fts_query)
    if city:
        query = query.filter(
            Address.city.ilike(city)
        )
    if state:
        query = query.filter(
            Address.state.ilike(state)
        )
    if zip_code:
        query = query.filter(
            Address.zip == zip_code
        )
    return query.count()

def get_address_by_id(
    db,
    address_id
):
    return (
        db.query(Address)
        .filter(
            Address.id == address_id
        )
        .first()
    )

def soft_delete_address(
    db,
    address_id
):
    address = (
        db.query(Address)
        .filter(
            Address.id == address_id
        )
        .first()
    )
    if not address:
        return None
    address.deleted_at = datetime.utcnow()
    db.commit()
    db.refresh(address)
    return address

def get_all_addresses(db):
    return (
        db.query(Address)
        .filter(Address.deleted_at == None)
        .all()
    )

def create_duplicate_candidate(
    db,
    address1_id,
    address2_id,
    score,
    recommendation="manual_review"
):
    existing = (
        db.query(DuplicateCandidate)
        .filter(
            DuplicateCandidate.address1_id == address1_id,
            DuplicateCandidate.address2_id == address2_id
        )
        .first()
    )
    if existing:
        if existing.recommendation == "manual_review" and recommendation != "manual_review":
            existing.recommendation = recommendation
            db.commit()
            db.refresh(existing)
        return existing
    candidate = DuplicateCandidate(
        address1_id=address1_id,
        address2_id=address2_id,
        score=score,
        status="pending",
        recommendation=recommendation
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return candidate

def get_pending_duplicate_candidates(db):
    from sqlalchemy.orm import joinedload
    return (
        db.query(DuplicateCandidate)
        .filter(
            DuplicateCandidate.status == "pending"
        )
        .options(
            joinedload(DuplicateCandidate.address1),
            joinedload(DuplicateCandidate.address2)
        )
        .all()
    )

def get_total_documents(db):
    return db.query(Document).count()

def get_unique_addresses(db):
    return (
        db.query(Address)
        .filter(Address.deleted_at == None)
        .count()
    )

def get_duplicate_addresses_count(db):
    total_links = db.query(AddressDocument).count()
    unique_addresses = (
        db.query(Address)
        .count()
    )
    return max(
        0,
        total_links - unique_addresses
    )

def update_address(
    db,
    address_id,
    street=None,
    city=None,
    state=None,
    zip_code=None
):
    from app.services.address_service import normalize_address
    address = (
        db.query(Address)
        .filter(Address.id == address_id)
        .first()
    )
    if not address:
        return None
    if street is not None:
        address.street = street
    if city is not None:
        address.city = city
    if state is not None:
        address.state = state
    if zip_code is not None:
        address.zip = zip_code
    current_street = address.street or ""
    current_city = address.city or ""
    current_state = address.state or ""
    current_zip = address.zip or ""
    reconstructed = f"{current_street}, {current_city}, {current_state} {current_zip}"
    address.normalized = normalize_address(reconstructed)
    address.review_status = "verified"
    db.commit()
    db.refresh(address)
    return address

def get_duplicate_candidate_by_id(
    db,
    candidate_id
):
    return (
        db.query(DuplicateCandidate)
        .filter(
            DuplicateCandidate.id == candidate_id
        )
        .first()
    )

def mark_candidate_not_duplicate(
    db,
    candidate
):
    candidate.status = "not_duplicate"
    db.commit()
    return candidate

def merge_addresses(
    db,
    winner_id,
    loser_id
):
    links = (
        db.query(AddressDocument)
        .filter(
            AddressDocument.address_id == loser_id
        )
        .all()
    )
    for link in links:
        existing = (
            db.query(AddressDocument)
            .filter(
                AddressDocument.address_id == winner_id,
                AddressDocument.document_id == link.document_id
            )
            .first()
        )
        if not existing:
            link.address_id = winner_id
    loser = (
        db.query(Address)
        .filter(Address.id == loser_id)
        .first()
    )
    if loser:
        loser.review_status = "merged"
    db.commit()
    return loser

def get_export_addresses(
    db,
    search=None,
    city=None,
    state=None,
    zip_code=None
):
    query = (
        db.query(Address)
        .filter(Address.deleted_at == None)
        .filter(Address.review_status != "merged")
    )
    if search:
        words = [w for w in search.strip().split() if w]
        if words:
            escaped_words = [w.replace('"', '""') for w in words]
            fts_query = " AND ".join(f'"{w}"*' for w in escaped_words)
            query = query.filter(
                text("addresses.id IN (SELECT id FROM addresses_fts WHERE addresses_fts MATCH :fts_query)")
            ).params(fts_query=fts_query)
    if city:
        query = query.filter(
            Address.city.ilike(city)
        )
    if state:
        query = query.filter(
            Address.state.ilike(state)
        )
    if zip_code:
        query = query.filter(
            Address.zip == zip_code
        )
    return query.all()

def get_document_by_content_hash(
    db,
    content_hash
):
    return (
        db.query(Document)
        .filter(
            Document.content_hash ==
            content_hash
        )
        .first()
    )

def get_total_addresses(db):
    return (
        db.query(Address)
        .filter(Address.deleted_at == None)
        .count()
    )

def get_total_duplicates(db):
    return (
        db.query(DuplicateCandidate)
        .filter(
            DuplicateCandidate.status == "pending"
        )
        .count()
    )

def create_rag_log(db, question: str, retrieved_files: str, latency_seconds: float):
    log = RagLog(
        question=question,
        retrieved_files=retrieved_files,
        latency_seconds=latency_seconds
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

