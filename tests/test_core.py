import numpy as np
import pytest
from backend.core import analyze, chunks, extract_skills, build_prompt, validate_text

class FakeEncoder:
    def encode(self, texts):
        # Deterministic test embeddings, no model download.
        output = []
        for text in texts:
            lower = text.lower()
            output.append([float('python' in lower), float('java' in lower), float('docker' in lower), 0.1])
        return np.array(output, dtype=np.float32)

def test_chunks_overlap():
    assert chunks('a b c d e f', max_words=4, overlap=2) == ['a b c d', 'c d e f']

def test_invalid_chunks():
    with pytest.raises(ValueError):
        chunks('hello', max_words=4, overlap=4)

def test_skill_boundaries():
    assert 'java' in extract_skills('Java and Python')
    assert 'java' not in extract_skills('javascript only')

def test_validation():
    with pytest.raises(ValueError):
        validate_text('', 'job')
    with pytest.raises(ValueError):
        validate_text('x' * 50001, 'job')

def test_retrieval_only_resume_and_bounded():
    result = analyze('Python developer experienced in Docker containers.', 'Python and Java required.', FakeEncoder())
    assert 'python' in result.matched_skills
    assert 'java' in result.missing_skills
    assert result.evidence
    assert all('resume_evidence' in x for x in result.evidence)
    assert 0 <= result.similarity <= 1
    assert '[R1:C1]' in build_prompt(result)

def test_invalid_top_k():
    with pytest.raises(ValueError):
        analyze('python', 'java', FakeEncoder(), top_k=0)
