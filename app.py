"""Streamlit frontend. Start the API first: uvicorn backend.api:app --host 127.0.0.1 --port 8000"""
import os
import requests
import streamlit as st

API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000')
MAX_UPLOAD = 5 * 1024 * 1024
st.set_page_config(page_title='ResumeMatch AI', page_icon='🧠', layout='wide')
st.title('ResumeMatch AI')
st.caption('Local-first resume coaching using semantic embeddings, FAISS retrieval and an optional LLM. Not a hiring decision tool.')


def extract_text(upload):
    if upload is None:
        return ''
    data = upload.getvalue()
    if len(data) > MAX_UPLOAD:
        raise ValueError('File exceeds 5 MB limit')
    if upload.name.lower().endswith('.txt'):
        return data.decode('utf-8', errors='replace')
    if not data.startswith(b'%PDF-'):
        raise ValueError('Invalid PDF header')
    import fitz
    with fitz.open(stream=data, filetype='pdf') as pdf:
        if pdf.page_count > 20:
            raise ValueError('PDF must be at most 20 pages')
        if pdf.needs_pass:
            raise ValueError('Password-protected PDF is not supported')
        return '\n'.join(page.get_text(sort=True) for page in pdf)

with st.sidebar:
    st.subheader('Settings')
    generate_report = st.checkbox('Generate LLM coaching report', value=True)
    top_k = st.slider('Evidence chunks per requirement', 1, 5, 3)
    st.info('Run locally. No documents are intentionally persisted by the app. Avoid public deployment without authentication.')

left, right = st.columns(2)
with left:
    upload = st.file_uploader('Resume (PDF/TXT, up to 5 MB)', type=['pdf', 'txt'])
    pasted = st.text_area('Or paste resume text', height=220)
with right:
    job = st.text_area('Job description', height=280)

if st.button('Analyze resume', type='primary'):
    try:
        resume = pasted.strip() or extract_text(upload).strip()
        if not resume or not job.strip():
            st.error('Provide both a readable resume and a job description.')
            st.stop()
        with st.spinner('Analyzing text with local embeddings...'):
            response = requests.post(API_URL + '/api/v1/analyze', json={
                'resume': resume, 'job': job, 'top_k': top_k, 'generate_report': generate_report
            }, timeout=(5, 240))
            response.raise_for_status()
            result = response.json()
        st.metric('Semantic similarity (not a fit %)', f"{result['similarity']:.3f}")
        st.caption(result['note'])
        a, b = st.columns(2)
        with a:
            st.subheader('Matching skill keywords')
            st.write(', '.join(result['matched_skills']) or 'None detected')
        with b:
            st.subheader('Unmatched job keywords')
            st.write(', '.join(result['missing_skills']) or 'None detected')
        with st.expander('Retrieved evidence', expanded=True):
            for item in result['evidence']:
                st.markdown(f"**[R{item['requirement_id']}:C{item['resume_chunk_id']}]** · similarity {item['similarity']:.3f}")
                st.write('Requirement:', item['requirement'])
                st.write('Resume evidence:', item['resume_evidence'])
        if result['report']:
            st.subheader('AI coaching report')
            st.markdown(result['report'])
            st.download_button('Download report', result['report'], file_name='resume_coaching.txt')
        elif result['report_status'] == 'ollama_unavailable':
            st.warning('Ollama is unavailable. Semantic matching and retrieval completed successfully.')
    except requests.HTTPError as exc:
        st.error(f'API error ({exc.response.status_code}): {exc.response.text[:400]}')
    except requests.RequestException:
        st.error('Cannot reach local API. Start uvicorn backend.api:app --host 127.0.0.1 --port 8000')
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
