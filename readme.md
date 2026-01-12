# Commands to run the app
### always do source /home/batman/Public/ChatPDF/.venv/bin/activate or whereever you are having the project codebase at
1. uv run uvicorn main:app
2. streamlit run streamlit_app.py
3. npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery