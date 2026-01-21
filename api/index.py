import base64
import hashlib
import hmac
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from openai import OpenAI

app = FastAPI(title="Legal Doc Processor (MVP)")

# -------------------------
# Config (env vars)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WIX_HMAC_SECRET = os.getenv("WIX_HMAC_SECRET", "")
ALLOWED_SKEW_SECONDS = int(os.getenv("ALLOWED_SKEW_SECONDS", "300"))

# Storage (S3-compatible: AWS S3 or Cloudflare R2)
STORAGE_ENDPOINT = os.getenv("STORAGE_ENDPOINT", "")  # e.g. https://<accountid>.r2.cloudflarestorage.com
STORAGE_REGION = os.getenv("STORAGE_REGION", "auto")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "")
STORAGE_ACCESS_KEY = os.getenv("STORAGE_ACCESS_KEY", "")
STORAGE_SECRET_KEY = os.getenv("STORAGE_SECRET_KEY", "")

# Controls
MAX_PAGES_TRIAL = int(os.getenv("MAX_PAGES_TRIAL", "30"))
MAX_PAGES_PLAN = int(os.getenv("MAX_PAGES_PLAN", "60"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "120000"))  # hard cap after extraction
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")  # change if needed

if not OPENAI_API_KEY:
    # Don’t crash import-time; raise on request if missing
    pass

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------
# Request/Response models
# -------------------------
class ProcessRequest(BaseModel):
    docId: str = Field(..., min_length=3)
    userId: str = Field(..., min_length=3)
    fileUrl: str = Field(..., min_length=10)
    fileName: str = Field(..., min_length=1)
    mode: str = Field(..., pattern="^(trial|active)$")
    language: str = Field(default="pt-BR")


class ProcessResponse(BaseModel):
    ok: bool
    docId: str
    result: Dict[str, Any]
    pdfUrl: str


# -------------------------
# Helpers
# -------------------------
def _now_ts() -> int:
    return int(time.time())


def verify_hmac(doc_id: str, user_id: str, timestamp: str, signature: str) -> None:
    if not WIX_HMAC_SECRET:
        raise HTTPException(status_code=500, detail="Server not configured: missing WIX_HMAC_SECRET")

    try:
        ts = int(timestamp)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    if abs(_now_ts() - ts) > ALLOWED_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Timestamp out of range")

    msg = f"{doc_id}|{user_id}|{timestamp}".encode("utf-8")
    expected = hmac.new(WIX_HMAC_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")


def sanitize_text(text: str) -> str:
    # Keep it readable and compact
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    num_pages = len(reader.pages)
    pages_to_read = min(num_pages, max_pages)

    parts = []
    for i in range(pages_to_read):
        page = reader.pages[i]
        parts.append(page.extract_text() or "")
    text = "\n\n".join(parts)
    text = sanitize_text(text)

    # Detect scanned PDF (very low text)
    if len(text) < 300:
        raise HTTPException(
            status_code=422,
            detail="O documento parece digitalizado (imagem) ou sem texto pesquisável. Envie um PDF pesquisável."
        )

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[TRUNCADO POR LIMITE DO MVP]"
    return text


def build_prompt(document_text: str, language: str) -> str:
    # IMPORTANT: triage/extraction only, no legal advice.
    return f"""
Você é um assistente de TRIAGEM e EXTRAÇÃO de informações de documentos jurídicos.
Você NÃO deve emitir parecer jurídico, opinião jurídica, recomendação estratégica ou conclusões definitivas.
Você deve APENAS extrair, organizar e resumir o conteúdo.

Idioma de saída: {language}

Retorne SOMENTE JSON válido (sem markdown), seguindo exatamente esta estrutura:

{{
  "document_type": "string",
  "confidence": 0.0,
  "parties": [{{"name":"string","role":"string","id_number":"string|null"}}],
  "object": "string",
  "key_dates": [{{"label":"string","date":"YYYY-MM-DD|null","raw":"string"}}],
  "amounts": [{{"label":"string","amount":"string","currency":"string|null","raw":"string"}}],
  "obligations": ["string"],
  "risks_points_of_attention": ["string"],
  "summary": "string",
  "checklist": ["string"],
  "missing_info": ["string"],
  "disclaimer": "Relatório automatizado para triagem e extração de informações. Não constitui parecer jurídico."
}}

Regras:
- Se não encontrar algo, use listas vazias e strings vazias; não invente.
- Datas: se conseguir, normalize para YYYY-MM-DD; senão use null e preencha "raw".
- Não cite jurisprudência nem “aconselhe”.
- Use termos jurídicos comuns, mas linguagem clara.

DOCUMENTO (texto extraído):
\"\"\"{document_text}\"\"\"
""".strip()


def call_openai_for_json(prompt: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server not configured: missing OPENAI_API_KEY")

    # Using Responses API (recommended by OpenAI)
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )
    output_text = getattr(resp, "output_text", None) or ""
    output_text = output_text.strip()

    # Try parse JSON safely
    try:
        data = json.loads(output_text)
        if not isinstance(data, dict):
            raise ValueError("JSON root is not object")
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {str(e)}")


def generate_pdf_report(result: Dict[str, Any], file_name: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def draw_title(txt: str, y: float) -> float:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, txt)
        return y - 20

    def draw_label_value(label: str, value: str, y: float) -> float:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(130, y, (value or "")[:120])
        return y - 14

    def draw_bullets(title: str, items: list, y: float) -> float:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, title)
        y -= 16
        c.setFont("Helvetica", 10)
        for it in items[:30]:
            line = str(it)
            c.drawString(50, y, f"• {line[:140]}")
            y -= 13
            if y < 70:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
        return y - 6

    # Cover-ish
    y = height - 50
    y = draw_title("Relatório de Triagem e Extração", y)
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Arquivo: {file_name}")
    y -= 14
    c.drawString(40, y, f"Gerado em: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')}")
    y -= 24

    y = draw_label_value("Tipo", str(result.get("document_type", "")), y)
    y = draw_label_value("Confiança", str(result.get("confidence", "")), y)
    y -= 10

    parties = result.get("parties", []) or []
    party_lines = []
    for p in parties[:10]:
        name = (p or {}).get("name", "")
        role = (p or {}).get("role", "")
        party_lines.append(f"{name} — {role}".strip(" —"))
    y = draw_bullets("Partes", party_lines, y)

    y = draw_bullets("Resumo", [result.get("summary", "")], y)
    y = draw_bullets("Obrigações", result.get("obligations", []) or [], y)
    y = draw_bullets("Pontos de atenção / riscos", result.get("risks_points_of_attention", []) or [], y)
    y = draw_bullets("Checklist", result.get("checklist", []) or [], y)

    missing = result.get("missing_info", []) or []
    if missing:
        y = draw_bullets("Informações ausentes", missing, y)

    disclaimer = result.get("disclaimer", "")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 40, disclaimer[:180])

    c.showPage()
    c.save()
    return buf.getvalue()


def s3_client():
    if not (STORAGE_BUCKET and STORAGE_ACCESS_KEY and STORAGE_SECRET_KEY):
        raise HTTPException(status_code=500, detail="Storage not configured (bucket/keys missing)")
    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name=STORAGE_REGION,
        aws_access_key_id=STORAGE_ACCESS_KEY,
        aws_secret_access_key=STORAGE_SECRET_KEY,
        endpoint_url=STORAGE_ENDPOINT or None,
    )


def upload_pdf_and_get_signed_url(pdf_bytes: bytes, key: str, expires_seconds: int = 86400) -> str:
    s3 = s3_client()
    s3.put_object(
        Bucket=STORAGE_BUCKET,
        Key=key,
        Body=pdf_bytes,
        ContentType="application/pdf",
        ACL="private",
    )
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": STORAGE_BUCKET, "Key": key},
        ExpiresIn=expires_seconds,
    )
    return url


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process", response_model=ProcessResponse)
async def process(
    req: ProcessRequest,
    x_signature: Optional[str] = Header(default=None),
    x_timestamp: Optional[str] = Header(default=None),
):
    # Auth
    verify_hmac(req.docId, req.userId, x_timestamp or "", x_signature or "")

    # Download PDF
    max_pages = MAX_PAGES_TRIAL if req.mode == "trial" else MAX_PAGES_PLAN

    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.get(req.fileUrl)
        if r.status_code != 200:
            raise HTTPException(status_code=422, detail="Não foi possível baixar o arquivo do Wix.")
        pdf_bytes = r.content

    # Extract text
    text = extract_text_from_pdf(pdf_bytes, max_pages=max_pages)

    # GPT to JSON
    prompt = build_prompt(text, req.language)
    result = call_openai_for_json(prompt)

    # Generate PDF
    safe_doc = re.sub(r"[^a-zA-Z0-9._-]+", "_", req.docId)
    pdf_key = f"reports/{req.userId}/{safe_doc}.pdf"
    pdf_bytes_out = generate_pdf_report(result, req.fileName)

    # Upload + signed URL
    pdf_url = upload_pdf_and_get_signed_url(pdf_bytes_out, pdf_key)

    return {"ok": True, "docId": req.docId, "result": result, "pdfUrl": pdf_url}
