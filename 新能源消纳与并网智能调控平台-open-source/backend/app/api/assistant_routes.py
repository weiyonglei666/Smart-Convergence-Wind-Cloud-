import base64
import io

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.adapters.stations import StationRepository
from app.api.routes import get_repository
from app.assistant.service import _get_index, answer_question

router = APIRouter(prefix="/api/assistant")


class AssistantRequest(BaseModel):
    # The question can include a parsed attachment excerpt. Keep the limit
    # high enough for a document context while still bounding request size.
    question: str = Field(min_length=1, max_length=18000)
    station_id: str | None = None
    use_web_search: bool = False
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


class AttachmentRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=240)
    mime_type: str = "application/octet-stream"
    content_base64: str = Field(min_length=1, max_length=14_000_000)


@router.get("/status")
def assistant_status():
    try:
        _, documents = _get_index()
        document_count = len(documents)
    except Exception as exc:
        return {"ready": False, "document_count": 0, "reason": str(exc)}
    from app.rag_config import model_configured
    return {"ready": True, "document_count": document_count, "model_configured": model_configured(), "agents": ["问题理解智能体", "实时数据智能体", "知识检索智能体", "报告生成智能体"]}


@router.post("/chat")
def assistant_chat(payload: AssistantRequest, repo: StationRepository = Depends(get_repository)):
    result = answer_question(payload.question, repo, payload.station_id, payload.use_web_search, payload.history)
    return {"answer": result.answer, "sources": result.sources, "steps": result.steps, "live_context": result.live_context, "mode": result.mode, "confidence": result.confidence}


@router.post("/attachment")
def parse_attachment(payload: AttachmentRequest):
    """Parse an uploaded document and return the text attached to the next prompt."""
    try:
        raw = base64.b64decode(payload.content_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="文件编码无效") from exc
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件不能超过 10 MB")
    suffix = payload.filename.lower().rsplit(".", 1)[-1] if "." in payload.filename else ""
    try:
        if suffix == "pdf" or payload.mime_type == "application/pdf":
            from pypdf import PdfReader
            text = "\n\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages).strip()
        elif suffix in {"txt", "md", "csv", "json"}:
            text = raw.decode("utf-8-sig", errors="replace").strip()
        else:
            raise HTTPException(status_code=415, detail="支持 PDF、TXT、MD、CSV、JSON 文件")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"文件解析失败：{exc}") from exc
    if not text:
        raise HTTPException(status_code=422, detail="文件没有可提取的文字内容（可能是扫描图片 PDF）")
    return {"filename": payload.filename, "characters": len(text), "text": text[:12000]}
