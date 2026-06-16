from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Float
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    size_bytes = Column(BigInteger)
    sha256 = Column(String, unique=True, index=True)
    status = Column(String)
    failure_reason = Column(String)
    doc_type = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    content_hash = Column(String, unique=True, nullable=True)
    content = Column(String, nullable=True)
    addresses = relationship(
        "AddressDocument",
        back_populates="document"
    )

class Address(Base):
    __tablename__ = "addresses"
    id = Column(Integer, primary_key=True, index=True)
    raw_text = Column(String)
    normalized = Column(String, index=True)
    street = Column(String)
    city = Column(String)
    state = Column(String)
    zip = Column(String)
    review_status = Column(String, default="unreviewed")
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    documents = relationship(
        "AddressDocument",
        back_populates="address"
    )

class AddressDocument(Base):
    __tablename__ = "address_documents"
    address_id = Column(
        Integer,
        ForeignKey("addresses.id"),
        primary_key=True
    )
    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        primary_key=True
    )
    address = relationship(
        "Address",
        back_populates="documents"
    )
    document = relationship(
        "Document",
        back_populates="addresses"
    )

class DuplicateCandidate(Base):
    __tablename__ = "duplicate_candidates"
    id = Column(Integer, primary_key=True)
    address1_id = Column(
        Integer,
        ForeignKey("addresses.id")
    )
    address2_id = Column(
        Integer,
        ForeignKey("addresses.id")
    )
    score = Column(Integer)
    status = Column(
        String,
        default="pending"
    )
    recommendation = Column(
        String,
        default="manual_review",
        nullable=True
    )
    address1 = relationship(
        "Address",
        foreign_keys=[address1_id],
        primaryjoin="DuplicateCandidate.address1_id==Address.id"
    )
    address2 = relationship(
        "Address",
        foreign_keys=[address2_id],
        primaryjoin="DuplicateCandidate.address2_id==Address.id"
    )

class RagLog(Base):
    __tablename__ = "rag_logs"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    retrieved_files = Column(String, nullable=False)
    latency_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

