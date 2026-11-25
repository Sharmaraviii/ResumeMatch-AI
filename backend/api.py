"""Local-first API. Do not expose publicly without authentication and TLS."""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.core import analyze, build_prompt, MAX_CHARS
from backend.services import LocalEncoder, ollama_generate

app = FastAPI(title='ResumeMatch AI', version='1.0.0', docs_url='/docs' if os.getenv('ENABLE_API_DOCS','1') == '1' else None)

class AnalysisRequest(BaseModel):
    resume: str = Field(min_length=1, max_length=MAX_CHARS)
    job: str = Field(min_length=1, max_length=MAX_CHARS)
    top_k: int = Field(default=3, ge=1, le=10)
    generate_report: bool = True

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/api/v1/analyze')
def analyze_endpoint(payload: AnalysisRequest):
    try:
        result = analyze(payload.resume, payload.job, LocalEncoder(), payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Embedding service unavailable') from exc
    output = result.as_dict()
    output['report'] = None
    output['report_status'] = 'disabled'
    if payload.generate_report:
        try:
            output['report'] = ollama_generate(build_prompt(result))
            output['report_status'] = 'generated'
        except RuntimeError:
            output['report_status'] = 'ollama_unavailable'
    return output
