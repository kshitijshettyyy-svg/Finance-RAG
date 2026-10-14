# FinWise RAG Assistant

A basic finance-only RAG assistant prototype for common people. It supports:

- Finance-only chat guardrails
- Groq API integration through `https://api.groq.com/openai/v1/chat/completions`
- PDF and text extraction for RAG context
- Image upload support for Grok multimodal input when `XAI_API_KEY` is set
- ITR filing explainer flow in the UI
- Local fallback responses for demos without an API key

## Run

```bash
npm install --cache ../.npm-cache
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

## Configure Groq

Create a `.env` file in this project folder or export these variables before running the server:

```bash
export GROQ_API_KEY="your_groq_api_key"
export GROQ_MODEL="openai/gpt-oss-120b"
```

The API key stays on the Node server and is not exposed to the browser. Do not paste the key into the frontend or share it in chat.

If you already put a Groq key in `XAI_API_KEY`, the prototype will still detect `gsk_` keys and use Groq automatically, but `GROQ_API_KEY` is clearer.

## Notes

This is a prototype, not financial, tax, investment, or legal advice. Final ITR filing decisions should be checked against official tax portal data or a qualified professional.
