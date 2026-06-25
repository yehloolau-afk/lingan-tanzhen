# Patent Scout · 灵感探针

**Live →** [yehloolau-afk.github.io/lingan-tanzhen](https://yehloolau-afk.github.io/lingan-tanzhen/)

A patent feasibility scanner for product ideas. Input a one-line description in Chinese, it searches CN/US/EP patent databases, scores your innovation, and returns 3 differentiated product directions — each with a mobile UI flow diagram.

Built for product managers and designers who want to validate ideas early, without knowing patent law.

---

## What you get

| Scenario | Output |
|---|---|
| Similar patents found | Top 1–5 matches in plain language, key differences from your idea |
| No similar patents | Innovation score (1–10), overall assessment |
| Novel idea | 3 product directions × mobile UI mockups (Ant Design Mobile) |

---

## Data sources

| Source | Coverage | Cost |
|---|---|---|
| Lens.org API | 95+ patent offices worldwide, including CNIPA | Free (registration required) |
| USPTO PatentsView | US patents, full database | Completely free, no key needed |
| DeepSeek API | Keyword extraction + analysis + report generation | Pay per use |

Search strategy: CN patents first → US/EP international patents. Chinese results are prioritized.

---

## How it works

- DeepSeek extracts Chinese and English search keywords from your idea
- Three parallel searches: Lens.org CN, Lens.org US/EP, USPTO PatentsView
- DeepSeek analyzes similarity and generates a report in Chinese
- If the idea is novel: 3 product directions are generated, each with a dedicated mobile UI mockup call

```
Your idea (Chinese)
    ↓
DeepSeek: extract keywords
    ↓
Parallel search (3 paths)
  ├── Lens.org CN   → Chinese patents
  ├── Lens.org US/EP → International patents
  └── USPTO PatentsView → US patents (fallback)
    ↓
DeepSeek: analyze → generate report
    ↓
3 × parallel mockup generation (Ant Design Mobile)
    ↓
Web UI
```

---

## Stack

- Python + FastAPI backend with SSE streaming
- Single HTML file frontend — no framework
- DeepSeek API (OpenAI-compatible)
- Lens.org Patent API + USPTO PatentsView API
- Ant Design Mobile visual spec for generated mockups
- Run locally: `python app.py` — opens browser automatically

`Python` · `FastAPI` · `DeepSeek API` · `Lens.org` · `USPTO` · `Vanilla HTML / CSS / JS`

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Fill in DEEPSEEK_API_KEY (required)
# Fill in LENS_API_KEY (strongly recommended — enables CN patent search)

# 3. Run
python app.py
# Opens http://localhost:8000 automatically
```

---

Star this if it is useful to you.
