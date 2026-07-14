# Trade Strategy Browser Demo

This project now includes a simple browser-based demo that can run in GitHub Codespaces or another cloud Python environment.

## What this gives you
- A VS Code-like environment in the browser
- A Streamlit web app you can open from your phone
- A working demo of your strategy pipeline with synthetic market data

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501
```

## Run in GitHub Codespaces
1. Push this folder to GitHub.
2. Open the repository in GitHub Codespaces.
3. In the terminal, run:
```bash
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501 --server.address 0.0.0.0
```
4. Open the forwarded port in your browser.

> The demo uses synthetic data so it can run easily in a browser-based environment.
