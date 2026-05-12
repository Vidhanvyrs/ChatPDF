import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
import base64

load_dotenv()

st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="centered")

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    is_prod = os.getenv("INNGEST_EVENT_KEY") is not None or os.getenv("VERCEL_ENV") is not None
    return inngest.Inngest(app_id="rag_app", is_production=is_prod)

async def send_rag_ingest_event(pdf_bytes: bytes, file_name: str) -> None:
    client = get_inngest_client()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_base64": pdf_base64,
                "file_name": file_name,
                "source_id": file_name,
            },
        )
    )

def _get_backend_url() -> str:
    # URL of your deployed Vercel app
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("📄 ChatPDF - Ingest")
uploaded = st.file_uploader("Upload a PDF to start chatting", type=["pdf"])

if uploaded is not None:
    if st.button("Ingest PDF"):
        with st.spinner("Uploading to Inngest..."):
            pdf_bytes = uploaded.getvalue()
            asyncio.run(send_rag_ingest_event(pdf_bytes, uploaded.name))
            st.success(f"Ingestion triggered for: {uploaded.name}")
            st.info("Wait a few seconds for the AI to process it, then ask a question below.")

st.divider()
st.title("💬 Chat with your PDF")

with st.form("query_form"):
    question = st.text_input("Ask something about your document")
    top_k = st.slider("Context chunks", 1, 10, 5)
    submit = st.form_submit_button("Ask AI")

    if submit and question.strip():
        with st.spinner("Thinking..."):
            backend_url = _get_backend_url()
            try:
                response = requests.post(
                    f"{backend_url}/api/query",
                    json={"question": question, "top_k": top_k},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                st.subheader("Answer")
                st.write(data.get("answer"))
                
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.write(f"- {s}")
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")
                st.info(f"Make sure BACKEND_URL is set to your Vercel URL (currently: {backend_url})")
