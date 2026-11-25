from fastapi.testclient import TestClient
from backend import api

client = TestClient(api.app)

def test_health():
    assert client.get('/health').json() == {'status': 'ok'}

def test_invalid_request():
    assert client.post('/api/v1/analyze', json={'resume':'', 'job':'java'}).status_code == 422

def test_analyze_api(monkeypatch):
    from tests.test_core import FakeEncoder
    monkeypatch.setattr(api, 'LocalEncoder', FakeEncoder)
    response = client.post('/api/v1/analyze', json={
        'resume':'Python Docker experience', 'job':'Python Java Docker requirements', 'generate_report':False
    })
    assert response.status_code == 200
    data = response.json()
    assert data['report_status'] == 'disabled'
    assert 'java' in data['missing_skills']

def test_ollama_fallback(monkeypatch):
    from tests.test_core import FakeEncoder
    monkeypatch.setattr(api, 'LocalEncoder', FakeEncoder)
    def unavailable(_):
        raise RuntimeError('offline')
    monkeypatch.setattr(api, 'ollama_generate', unavailable)
    response = client.post('/api/v1/analyze', json={
        'resume':'Python', 'job':'Java', 'generate_report':True
    })
    assert response.status_code == 200
    assert response.json()['report_status'] == 'ollama_unavailable'
