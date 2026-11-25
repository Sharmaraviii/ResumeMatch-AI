# Evaluation protocol

1. Create at least 30 synthetic or consented resume/job pairs with annotated relevant resume spans for each job requirement.
2. Evaluate FAISS retrieval against those labels using Precision@K, Recall@K, and MRR. Compare with a keyword baseline.
3. Manually audit at least 30 generated coaching claims for source support and correct evidence citations; report unsupported-claim rate.
4. Benchmark p50/p95 latency for embedding/retrieval and LLM generation on your actual laptop, noting CPU/GPU and memory.
5. Check long PDFs, scanned PDFs, adversarial instructions in resumes, missing qualifications, and empty input.
6. Record methodology, raw counts, failures, and hardware before making any quantitative resume claims.

Do not treat semantic cosine similarity as hiring suitability or evaluate candidates' protected traits.
