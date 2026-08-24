# YouTube Transcript RAG Assistant

## Run locally
1. Copy `.env.example` to `.env` and put your real key in it:
   ```
   cp .env.example .env
   ```
   `.env` should contain:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```
2. Install deps and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
`.env` is already in `.gitignore` so it won't get pushed to GitHub.

## Deploy on Streamlit Cloud
`.env` files are NOT uploaded when you push to GitHub (gitignored on purpose — never commit your key). Streamlit Cloud doesn't read `.env` either. Instead:
1. Push this folder to a GitHub repo (app.py + requirements.txt at root).
2. Go to share.streamlit.io -> New app -> pick repo/branch -> main file path: `app.py`.
3. In the app's dashboard: Settings -> Secrets, add:
   ```
   GROQ_API_KEY = "your_actual_key_here"
   ```
4. Deploy.

The app currently reads `os.getenv("GROQ_API_KEY")`, which works locally via `.env`. If you want it to also pick up Streamlit Cloud's secrets automatically without extra setup, let me know and I'll add a one-line fallback to `st.secrets`.

## Notes
- If `GROQ_API_KEY` isn't found in `.env`, the app falls back to a sidebar text input so it never fully breaks.
- FAISS index is built fresh per video URL via `st.cache_resource`, so repeat questions on the same video are fast.
- `qwen/qwen3.6-27b` must be an active model on your Groq account — check Groq's model list if you get a 404/model error.
