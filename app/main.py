"""FastAPI backend entrypoint for the AI Utility Agent."""

import json
import logging

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent import agent
from app.config import settings, configure_logging
from app.models import (
    ChatRequest,
    ChatResponse,
    CalculatorRequest,
    WeatherRequest,
    SearchRequest,
    WikipediaRequest,
    PDFChatRequest,
    PDFDeleteRequest,
)
from app.rag.manager import rag_manager
from app.tools.registry import TOOLS

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A multi-tool AI agent: Gemini chat, PDF RAG chat, calculator, weather, web search and Wikipedia.",
)

# Allow the Streamlit frontend (local or deployed) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so the API never returns a raw 500 stack trace to clients."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {exc}"})


@app.get("/")
def home():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "Running",
        "tools": list(TOOLS.keys()),
        "max_pdfs": settings.MAX_PDFS,
        "max_pdf_size_mb": settings.MAX_PDF_SIZE_MB,
        "groq_configured": bool(settings.GROQ_API_KEY),
    }


@app.get("/health")
def health():
    return {"status": "Healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    """Main entrypoint: the agent decides which tool (if any) should answer."""
    result = agent.run(data.message)
    return result


@app.post("/chat/stream")
def chat_stream(data: ChatRequest):
    """Same routing as /chat, but streams the response as newline-delimited
    JSON events so the UI can render text as it arrives (ChatGPT-style).
    Structured tools (calculator/weather/wikipedia/search) still resolve
    instantly and are sent as a single 'result' event.
    """

    def event_gen():
        try:
            for event in agent.stream(data.message):
                yield json.dumps(event) + "\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")


@app.post("/calculator")
def calculator(data: CalculatorRequest):
    try:
        result = TOOLS["calculator"].run(
            num1=data.num1, num2=data.num2, operation=data.operation
        )
        return {"result": result}
    except Exception as e:  # noqa: BLE001
        logger.warning("Calculator error: %s", e)
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/weather")
def weather(data: WeatherRequest):
    result = TOOLS["weather"].run(data.city)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/search")
def search(data: SearchRequest):
    result = TOOLS["search"].run(data.query)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/wikipedia")
def wikipedia_lookup(data: WikipediaRequest):
    result = TOOLS["wikipedia"].run(data.query)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


# ---------------------------------------------------------------------
# PDF RAG endpoints
# ---------------------------------------------------------------------
@app.post("/pdf-chat")
def pdf_chat(data: PDFChatRequest):
    """Non-streaming PDF question answering (used by /chat too, via the pdf tool)."""
    return rag_manager.ask(data.message)


@app.post("/pdf-chat/stream")
def pdf_chat_stream(data: PDFChatRequest):
    """Streaming PDF question answering - text chunks followed by a sources event."""

    def event_gen():
        try:
            for chunk in rag_manager.ask_stream(data.message):
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"
            yield json.dumps({"type": "sources", "sources": rag_manager.last_sources()}) + "\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("Streaming PDF chat failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")


@app.post("/pdf/upload")
async def pdf_upload(files: list[UploadFile] = File(...)):
    """Upload one or more PDFs. Automatically (re)builds the vector index."""
    if len(rag_manager.files) + len(files) > settings.MAX_PDFS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Maximum of {settings.MAX_PDFS} PDFs allowed."},
        )

    uploads = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400, content={"error": f"'{f.filename}' is not a PDF file."}
            )

        content = await f.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.MAX_PDF_SIZE_MB:
            return JSONResponse(
                status_code=400,
                content={"error": f"'{f.filename}' exceeds the {settings.MAX_PDF_SIZE_MB}MB limit."},
            )
        uploads.append((f.filename, content))

    try:
        result = rag_manager.add_pdfs(uploads)
        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("PDF upload/indexing failed")
        return JSONResponse(status_code=500, content={"error": f"Failed to process PDF(s): {e}"})


@app.get("/pdf/list")
def pdf_list():
    return {"files": rag_manager.list_pdfs()}


@app.post("/pdf/delete")
def pdf_delete(data: PDFDeleteRequest):
    ok = rag_manager.remove_pdf(data.filename)
    if not ok:
        return JSONResponse(status_code=404, content={"error": f"'{data.filename}' not found."})
    return {"status": "deleted", "filename": data.filename, "total_files": len(rag_manager.files)}


@app.post("/pdf/rebuild")
def pdf_rebuild():
    try:
        return rag_manager.rebuild()
    except Exception as e:  # noqa: BLE001
        logger.exception("PDF index rebuild failed")
        return JSONResponse(status_code=500, content={"error": f"Rebuild failed: {e}"})


@app.post("/pdf/clear")
def pdf_clear():
    rag_manager.clear_all()
    return {"status": "cleared"}
