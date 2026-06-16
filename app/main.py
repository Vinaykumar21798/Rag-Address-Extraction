import os
import csv
import re
import time
from io import StringIO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from app.logger import logger
from app.exceptions import LLMUnavailable
from app.schemas import SearchRequest
from app.database.db import (
    init_db,
    get_db,
    create_document,
    get_documents,
    get_document_by_id,
    get_document_by_sha256,
    get_document_by_content_hash,
    create_address,
    get_address_by_normalized,
    create_address_document_link,
    get_addresses,
    get_addresses_count,
    get_address_by_id,
    soft_delete_address,
    get_pending_duplicate_candidates,
    get_all_addresses,
    create_duplicate_candidate,
    get_duplicate_candidate_by_id,
    mark_candidate_not_duplicate,
    merge_addresses,
    update_address,
    get_export_addresses,
    get_total_documents,
    get_total_addresses,
    get_total_duplicates,
    get_unique_addresses,
    get_duplicate_addresses_count,
)
from app.hash_utils import generate_sha256, generate_content_hash
from app.services.pdf_service import extract_text_from_pdf
from app.services.duplicate_detector import calculate_similarity
from app.services.recommendation_service import determine_recommendation
from app.extractor import extract_addresses
from app.regex_extractor import extract_addresses_regex
from app.ingestion import ingest_document
from app.vector_store import collection, search
from app.query_rewriter import rewrite_query
from app.rag import answer_question, AskResponse
app = FastAPI(title="RAG Address Registry System")
CORPUS_DIR = Path("corpus")

class DocumentRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    question: str

class AddressUpdateRequest(BaseModel):
    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None

class DuplicateResolveRequest(BaseModel):
    action: str

@app.on_event("startup")

def startup_event():
    init_db()
    logger.info("Database initialized")

@app.get("/")

@app.get("/ask_ui")

def home():
    return FileResponse("templates/index.html")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        content = await file.read()
        if len(content) == 0:
            raise ValueError("Uploaded file is empty")
        file_hash = generate_sha256(content)
        logger.debug(f"Generated SHA256 hash: {file_hash}")
        existing_document = get_document_by_sha256(db, file_hash)
        if existing_document:
            logger.warning(f"Duplicate file rejected: {file.filename}")
            db.execute(
                text("INSERT OR REPLACE INTO rejected_duplicates (sha256) VALUES (:sha256)"),
                {"sha256": file_hash}
            )
            db.commit()
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Duplicate file",
                    "document_id": existing_document.id
                }
            )
        try:
            if file.filename.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(content)

            else:
                extracted_text = content.decode("utf-8", errors="ignore")
            if not extracted_text.strip():
                raise ValueError("No extractable text found in file")
            logger.info(f"Extracted text from {file.filename}")
            content_hash = generate_content_hash(extracted_text)
            existing_content = get_document_by_content_hash(db, content_hash)
            if existing_content:
                logger.warning(f"Content duplicate detected: {file.filename}")
                db.execute(
                    text("INSERT OR REPLACE INTO rejected_duplicates (sha256) VALUES (:sha256)"),
                    {"sha256": file_hash}
                )
                db.commit()
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Content duplicate",
                        "document_id": existing_content.id
                    }
                )
            regex_result = extract_addresses_regex(extracted_text)
            addresses_found = regex_result.get("addresses", [])

        except HTTPException:
            raise

        except Exception as e:
            document = create_document(
                db=db,
                filename=file.filename,
                size_bytes=len(content),
                sha256=file_hash,
                status="failed",
                failure_reason=str(e),
                content=""
            )
            logger.error(f"Upload failed: {file.filename} - {str(e)}")
            return {
                "document_id": document.id,
                "status": "failed",
                "reason": str(e)
            }
        document = create_document(
            db=db,
            filename=file.filename,
            size_bytes=len(content),
            sha256=file_hash,
            content_hash=content_hash,
            status="processed",
            content=extracted_text
        )
        register_addresses_helper(db, document.id, addresses_found)
        ingest_document(file.filename, extracted_text)
        logger.info(f"Document uploaded and indexed: {file.filename}")
        return {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "addresses_found": len(addresses_found)
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Upload failed: {file.filename} - {str(e)}")
        try:
            document = create_document(
                db=db,
                filename=file.filename,
                size_bytes=0,
                status="failed",
                failure_reason=str(e),
                content=""
            )
            return {
                "document_id": document.id,
                "status": "failed",
                "reason": str(e)
            }

        except Exception:
            raise HTTPException(status_code=500, detail=str(e))

def register_addresses_helper(db: Session, doc_id: int, addresses_list: list[dict]):
    processed_normalized = set()
    all_addresses = get_all_addresses(db)
    for addr_dict in addresses_list:
        normalized_str = f"{addr_dict['street']}, {addr_dict['city']}, {addr_dict['state']} {addr_dict['zip']}".upper().strip()
        normalized_str = re.sub(r'\s+', ' ', normalized_str)
        if normalized_str in processed_normalized:
            continue
        processed_normalized.add(normalized_str)
        existing_address = get_address_by_normalized(db, normalized_str)
        if existing_address:
            create_address_document_link(
                db=db,
                address_id=existing_address.id,
                document_id=doc_id
            )
            continue
        raw_text = f"{addr_dict['street']}, {addr_dict['city']}, {addr_dict['state']} {addr_dict['zip']}"
        address = create_address(
            db=db,
            raw_text=raw_text,
            normalized=normalized_str,
            street=addr_dict["street"],
            city=addr_dict["city"],
            state=addr_dict["state"],
            zip_code=addr_dict["zip"]
        )
        for existing in all_addresses:
            if existing.id == address.id:
                continue
            same_city = (
                existing.city is not None and
                address.city is not None and
                existing.city.strip().upper() == address.city.strip().upper()
            )
            same_zip = (
                existing.zip is not None and
                address.zip is not None and
                existing.zip.strip() == address.zip.strip()
            )
            if not (same_city or same_zip):
                continue
            score = calculate_similarity(existing.normalized, address.normalized)
            if score >= 90:
                recommendation = determine_recommendation(existing, address)
                create_duplicate_candidate(
                    db=db,
                    address1_id=existing.id,
                    address2_id=address.id,
                    score=int(score),
                    recommendation=recommendation
                )
        create_address_document_link(
            db=db,
            address_id=address.id,
            document_id=doc_id
        )

@app.post("/ask", response_model=AskResponse)

def ask(request: AskRequest, db: Session = Depends(get_db)):
    try:
        return answer_question(request.question, db)

    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{document_id}/extract_llm")

def extract_llm_endpoint(document_id: int, db: Session = Depends(get_db)):
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.content:
        raise HTTPException(status_code=400, detail="Document has no content")
    try:
        result = extract_addresses(document.content)
        addresses_found = result.get("addresses", {}).get("addresses", [])
        register_addresses_helper(db, document.id, addresses_found)
        return result

    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_llm")

def extract_llm_raw(request: DocumentRequest):
    try:
        return extract_addresses(request.text)

    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/rag/reindex")

def reindex():
    total_chunks = 0
    try:
        ids_to_delete = collection.get()["ids"]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    except Exception:
        pass
    for file in CORPUS_DIR.glob("*"):
        if file.is_file():
            text_content = file.read_text(encoding="utf-8", errors="ignore")
            total_chunks += ingest_document(file.name, text_content)
    return {"chunks": total_chunks}

@app.get("/rag/stats")

def rag_stats():
    count = collection.count()
    return {"chunks": count}

@app.post("/rag/search")

def rag_search(request: SearchRequest):
    query_str = request.question
    if request.rewrite:
        try:
            query_str = rewrite_query(request.question)

        except Exception:
            pass
    results = search(query_str, request.k)
    hits = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        for doc, meta, distance in zip(docs, metas, distances):
            similarity = max(0, round(1 / (1 + distance), 4))
            hits.append({
                "filename": meta.get("filename") or meta.get("source") or "",
                "similarity": similarity,
                "text": doc
            })
    return {
        "question": request.question,
        "rewritten_question": query_str if request.rewrite else None,
        "results": hits
    }

@app.get("/documents")

def list_documents(db: Session = Depends(get_db)):
    documents = get_documents(db)
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "size_bytes": doc.size_bytes,
            "status": doc.status,
            "failure_reason": doc.failure_reason,
            "uploaded_at": doc.uploaded_at
        }
        for doc in documents
    ]

@app.get("/documents/{document_id}")

def get_document_endpoint(document_id: int, db: Session = Depends(get_db)):
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": document.id,
        "filename": document.filename,
        "size_bytes": document.size_bytes,
        "status": document.status,
        "failure_reason": document.failure_reason,
        "uploaded_at": document.uploaded_at
    }

@app.get("/addresses")

def list_addresses(
    limit: int = 20,
    offset: int = 0,
    search_query: str = None,
    city: str = None,
    state: str = None,
    zip_code: str = None,
    db: Session = Depends(get_db)
):
    addresses = get_addresses(
        db=db,
        limit=limit,
        offset=offset,
        search=search_query,
        city=city,
        state=state,
        zip_code=zip_code
    )
    total = get_addresses_count(
        db=db,
        search=search_query,
        city=city,
        state=state,
        zip_code=zip_code
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [
            {
                "id": address.id,
                "raw_text": address.raw_text,
                "normalized": address.normalized,
                "street": address.street,
                "city": address.city,
                "state": address.state,
                "zip": address.zip,
                "review_status": address.review_status
            }
            for address in addresses
        ]
    }

@app.get("/addresses/{address_id}")

def get_address_endpoint(address_id: int, db: Session = Depends(get_db)):
    address = get_address_by_id(db, address_id)
    if not address:
        raise HTTPException(status_code=404, detail="Address not found")
    return {
        "id": address.id,
        "raw_text": address.raw_text,
        "normalized": address.normalized,
        "street": address.street,
        "city": address.city,
        "state": address.state,
        "zip": address.zip,
        "review_status": address.review_status,
        "documents": [
            {
                "document_id": link.document.id,
                "filename": link.document.filename
            }
            for link in address.documents
        ]
    }

@app.delete("/addresses/{address_id}")

def delete_address_endpoint(address_id: int, db: Session = Depends(get_db)):
    address = soft_delete_address(db, address_id)
    if not address:
        raise HTTPException(status_code=404, detail="Address not found")
    return {
        "message": "Address soft deleted",
        "address_id": address.id
    }

@app.patch("/addresses/{address_id}")

def update_address_endpoint(
    address_id: int,
    request: AddressUpdateRequest,
    db: Session = Depends(get_db)
):
    address = update_address(
        db=db,
        address_id=address_id,
        street=request.street,
        city=request.city,
        state=request.state,
        zip_code=request.zip
    )
    if not address:
        raise HTTPException(status_code=404, detail="Address not found")
    return {
        "message": "Address updated",
        "address_id": address.id
    }

@app.get("/duplicates")

def get_duplicates_endpoint(db: Session = Depends(get_db)):
    candidates = get_pending_duplicate_candidates(db)
    result = []
    for candidate in candidates:
        address1 = get_address_by_id(db, candidate.address1_id)
        address2 = get_address_by_id(db, candidate.address2_id)
        result.append({
            "id": candidate.id,
            "address1_id": candidate.address1_id,
            "address2_id": candidate.address2_id,
            "address1_text": address1.normalized if address1 else "",
            "address2_text": address2.normalized if address2 else "",
            "score": candidate.score,
            "status": candidate.status,
            "recommendation": candidate.recommendation
        })
    return result

@app.post("/duplicates/{candidate_id}/resolve")

def resolve_duplicate(
    candidate_id: int,
    request: DuplicateResolveRequest,
    db: Session = Depends(get_db)
):
    candidate = get_duplicate_candidate_by_id(db, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if request.action == "not_duplicate":
        mark_candidate_not_duplicate(db, candidate)
        return {"message": "Marked as not duplicate"}
    if request.action == "merge":
        merge_addresses(
            db,
            winner_id=candidate.address1_id,
            loser_id=candidate.address2_id
        )
        candidate.status = "merged"
        db.commit()
        return {"message": "Addresses merged"}
    raise HTTPException(status_code=400, detail="Invalid action")

@app.get("/export")

def export_csv(
    search_query: str = None,
    city: str = None,
    state: str = None,
    zip_code: str = None,
    db: Session = Depends(get_db)
):
    addresses = get_export_addresses(
        db,
        search=search_query,
        city=city,
        state=state,
        zip_code=zip_code
    )
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "street", "city", "state", "zip", "normalized"])
    for address in addresses:
        writer.writerow([
            address.id,
            address.street,
            address.city,
            address.state,
            address.zip,
            address.normalized
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=addresses.csv"}
    )

@app.get("/stats")

@app.get("/statistics")

def get_stats_endpoint(db: Session = Depends(get_db)):
    total_docs = get_total_documents(db)
    unique_addrs = get_unique_addresses(db)
    try:
        dup_files_rejected = db.execute(text("SELECT COUNT(*) FROM rejected_duplicates")).scalar() or 0

    except Exception:
        dup_files_rejected = 0
    dup_addrs_caught = get_duplicate_addresses_count(db)
    return {
        "total_documents": total_docs,
        "unique_addresses": unique_addrs,
        "duplicate_files_rejected": dup_files_rejected,
        "duplicate_addresses_caught": dup_addrs_caught
    }
