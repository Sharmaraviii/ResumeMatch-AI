# ResumeMatch AI

**Local-first resume coaching with semantic ML embeddings, FAISS retrieval, and a RAG-powered LLM.** This is a portfolio-ready application, **not a validated automated hiring or applicant ranking system**.

## Features

- PDF/TXT resume upload or pasted resume; pasted job description
- Local Sentence Transformers `all-MiniLM-L6-v2` embeddings
- Cosine similarity over mean-pooled chunk embeddings (**not a job-fit percentage**)
- FAISS retrieval of resume evidence against job-description chunks (avoids self-matching)
- Transparent exact keyword skill overlap/gaps; not an exhaustive skill ontology
- Optional local Ollama `qwen2.5:3b` grounded coaching with traceable chunk labels
- FastAPI backend with OpenAPI docs and Streamlit dashboard
- File size/page limits, bounded requests, health endpoint, CI, tests, Docker Compose
- No database; document text is not intentionally stored by application code

## Architecture

```text
Browser -> Streamlit UI -> FastAPI /api/v1/analyze
                                |-> chunk + embed resume/job (MiniLM)
                                |-> normalized cosine similarity
                                |-> FAISS job-chunk -> resume-chunk retrieval
                                |-> skill keyword analysis
                                `-> optional Ollama/Qwen -> cited coaching report
```

## Local setup (recommended)

Python **3.11** recommended. Ollama requires a separate install and model download. All dependencies/models are free to use locally, but first-time downloads require internet, and local hardware/electricity have costs.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

Start Ollama (desktop app or `ollama serve`), then in **two terminals**:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
streamlit run app.py
```

Open `http://localhost:8501`. API docs: `http://127.0.0.1:8000/docs`. Use `sample_resume.txt` and `sample_job.txt` to demo. Disable the LLM checkbox if Ollama is not installed. The first analysis downloads the embedding model.

### Docker Compose (optional)

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:3b
```

Open `http://localhost:8501`. The embedding model downloads on first use in the API container. Docker Ollama may be slow on CPU; native local setup is simpler on many laptops. This compose configuration is for **local development**, not Internet exposure.

## API example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{"resume":"Python and Docker projects", "job":"Python and Java backend experience", "generate_report":false}'
```

Returns `similarity`, `matched_skills`, `missing_skills`, `evidence`, `note`, `report`, and `report_status`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests use a deterministic fake encoder; they do **not** prove the quality of the real embedding model or LLM. For full end-to-end verification, run the app with Ollama and manually compare retrieved passages to their sources.

## Evaluation before claiming performance

Create a labeled set of job requirements and relevant resume chunks, then calculate Precision@K/Recall@K. Also measure unsupported LLM claims and latency. **Do not invent accuracy percentages for your resume.** See [evaluation plan](docs/EVALUATION.md).

## Privacy, security and limitations

- This project is intended for **local single-user development**. There is no authentication, rate limiting, TLS, or secure multi-user isolation. **Do not expose it on a public server or process real applicant data without these controls and a privacy review.**
- Model downloads may contact Hugging Face; Ollama inference runs locally when configured as documented.
- PDF parser accepts text PDFs only; scanned PDFs require OCR (not included).
- LLMs can hallucinate or mishandle adversarial text despite prompt instructions; always verify citations.
- Semantic similarity is not a probability of candidate suitability; exact keyword detection misses synonyms and nuanced skills.
- No personal resumes should be committed to Git. Test data is synthetic.
- Docker/CI are provided but do not imply the system has passed production security, scalability, or load testing.

## GitHub publishing

1. Review source and sample data; never commit private resumes, API tokens, `.env`, model weights, or caches.
2. Replace this README's demo placeholders with **your own verified screenshots** after running the app.
3. Run tests locally, then `git init`, `git add .`, `git commit -m "Initial ResumeMatch AI release"`.
4. Create an empty GitHub repository called `resume-match-ai` and run the `git remote add origin ...` / `git push -u origin main` commands shown by GitHub. Ensure `git branch -M main` first.
5. Wait for the GitHub Actions CI check to pass. Add project description, topics (`rag`, `llm`, `nlp`, `faiss`, `fastapi`, `streamlit`) and a demo video.

## Resume bullet examples (use only once personally verified)

- Built a local-first resume analysis platform using Sentence Transformers, FAISS vector retrieval, and LLM-powered evidence-grounded coaching.
- Developed FastAPI endpoints and a Streamlit dashboard for semantic skill comparison, cited evidence retrieval, and explainable resume feedback.
- Added automated API/domain tests and GitHub Actions CI, with Docker Compose for reproducible local deployment.

## License

MIT. Check the individual model and dependency licenses for your intended use.
