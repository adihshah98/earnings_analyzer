"""RAG API routes for ingestion and search."""

import logging

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import require_admin_key

from app.ingestion.docx_parser import extract_text_from_docx
from app.models.schemas import (
    IngestResult,
    ManualIngestMissingResponse,
    SearchRequest,
    SourceDocument,
)
from app.rag.ingestion import ingest_document
from app.rag.retriever import _COMPANIES_CACHE, get_transcript_by_chunk_id, retrieve_by_doc_id, retrieve_relevant_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


def _missing_fields_response(missing: list[str]) -> dict:
    """Build the 422 detail dict for missing required fields."""
    return {
        "missing": missing,
        "message": (
            "To ingest this transcript, please provide: "
            + ", ".join(missing)
            + ". "
            + ("company_ticker is the stock symbol (e.g. AAPL, NOW). " if "company_ticker" in missing else "")
            + ("call_date is the date of the call in YYYY-MM-DD format. " if "call_date" in missing else "")
        ),
    }


@router.post(
    "/ingest/manual/upload",
    response_model=IngestResult,
    dependencies=[Depends(require_admin_key)],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Missing required fields (ticker, call_date) or invalid file",
            "model": ManualIngestMissingResponse,
        },
    },
)
async def ingest_manual_upload(
    file: UploadFile = File(..., description="Word document (.docx) containing the transcript"),
    company_ticker: str | None = Form(None, description="Company ticker (e.g. AAPL). Required."),
    call_date: str | None = Form(None, description="Date of the call, YYYY-MM-DD. Required."),
    title: str | None = Form(None, description="Optional title; if omitted, generated from ticker and date."),
    use_eval_table: bool = Form(False, description="If true, ingest into eval_document_chunks (for eval transcripts)."),
) -> IngestResult:
    """Upload a Word (.docx) transcript. Text is extracted automatically.

    You must also provide **company_ticker** and **call_date** as form fields.
    If either is missing, returns 422 with ``missing`` and ``message`` so the
    client can prompt the user. Accepts only .docx files.
    Set use_eval_table=true to store in eval_document_chunks (for eval seeding).
    """
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "missing": [],
                "message": "Only Word documents (.docx) are supported. Please upload a .docx file.",
            },
        )

    content_bytes = await file.read()
    try:
        content = extract_text_from_docx(content_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "missing": [],
                "message": str(e),
            },
        ) from e

    if len(content.strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "missing": [],
                "message": "The document contains too little text to ingest (need at least 10 characters).",
            },
        )

    missing: list[str] = []
    if not company_ticker or not company_ticker.strip():
        missing.append("company_ticker")
    if not call_date or not call_date.strip():
        missing.append("call_date")
    else:
        try:
            datetime.strptime(call_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "missing": ["call_date"],
                    "message": "call_date must be in YYYY-MM-DD format (e.g. 2024-10-30).",
                },
            )

    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_missing_fields_response(missing),
        )

    ticker = company_ticker.strip().upper()
    call_date_parsed = datetime.strptime(call_date.strip(), "%Y-%m-%d").date()
    title_final = (title.strip() if title and title.strip() else None) or f"{ticker} Manual Earnings Call {call_date_parsed}"

    result = await ingest_document(
        title=title_final,
        content=content,
        source="transcript",
        company_ticker=ticker,
        call_date=call_date_parsed,
        metadata={"data_source": "manual_upload", "original_filename": file.filename or ""},
        use_transcript_chunking=True,
        use_eval_table=use_eval_table,
    )
    _COMPANIES_CACHE.clear()
    return IngestResult(**result)


@router.post("/search", response_model=list[SourceDocument], dependencies=[Depends(require_admin_key)])
async def search_documents(request: SearchRequest):
    """Perform semantic search over the knowledge base."""
    filter_metadata: dict | None = None
    if request.company_ticker:
        filter_metadata = {"company_ticker": request.company_ticker}

    chunks = await retrieve_relevant_chunks(
        query=request.query,
        top_k=request.top_k,
        threshold=request.threshold,
        filter_metadata=filter_metadata,
        search_mode=request.search_mode,
    )

    return [
        SourceDocument(
            chunk_id=c["chunk_id"],
            content=c["content"],
            similarity=c["similarity"],
            metadata=c.get("metadata", {}),
        )
        for c in chunks
    ]


@router.get("/chunks/{chunk_id}/transcript")
async def get_transcript_for_chunk(chunk_id: str):
    """Return the full transcript (all chunks) for the document containing this chunk.

    Lets the frontend show the broader source when a user clicks a chunk.
    """
    data = await get_transcript_by_chunk_id(chunk_id)
    if not data:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return data


@router.get("/documents/{doc_id}/chunks", dependencies=[Depends(require_admin_key)])
async def get_document_chunks(doc_id: str, eval_table: bool = False):
    """Return all chunks for a document by doc_id. Used to debug ingestion.

    doc_id is returned by POST /rag/ingest/manual/upload and is computed
    as sha256(company_ticker:call_date).hexdigest()[:16].
    Set eval_table=true to read from eval_document_chunks.
    """
    chunks = await retrieve_by_doc_id(doc_id, use_eval_table=eval_table)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "chunks": chunks, "count": len(chunks)}


