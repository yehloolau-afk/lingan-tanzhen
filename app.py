import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yehloolau-afk.github.io", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LENS_API_KEY = os.getenv("LENS_API_KEY", "")
deepseek = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


class IdeaRequest(BaseModel):
    idea: str


def parse_json_safe(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    return json.loads(text)


async def extract_keywords(idea: str) -> dict:
    resp = await deepseek.chat.completions.create(
        model="deepseek-chat",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                f'用户有一个想法："{idea}"\n\n'
                "提取用于专利数据库检索的关键词。\n"
                "只返回JSON，不要其他文字：\n"
                '{"zh_keywords": ["核心词1", "核心词2", "核心词3"], '
                '"en_keywords": ["keyword1", "keyword2", "keyword3"], '
                '"search_query": "english boolean query for patent database", '
                '"tech_field": "技术领域（10字以内）"}'
            )
        }]
    )
    return parse_json_safe(resp.choices[0].message.content)


async def search_lens(query: str, jurisdiction: str, size: int = 10) -> list:
    if not LENS_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.lens.org/patent/search",
                headers={
                    "Authorization": f"Bearer {LENS_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": {
                        "bool": {
                            "must": [{
                                "query_string": {
                                    "query": query,
                                    "fields": ["title.text", "abstract.text"],
                                    "default_operator": "OR"
                                }
                            }],
                            "filter": [{"term": {"jurisdiction": jurisdiction.upper()}}]
                        }
                    },
                    "size": size,
                    "fields": ["lens_id", "jurisdiction", "publication_date", "biblio", "abstract"],
                    "sort": [{"publication_date": "desc"}]
                }
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            print(f"Lens {jurisdiction} error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"Lens {jurisdiction} exception: {e}")
    return []


async def search_uspto(query: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://search.patentsview.org/api/v1/patent/",
                params={
                    "q": json.dumps({"_text_any": {"patent_title": query, "patent_abstract": query}}),
                    "f": json.dumps(["patent_id", "patent_title", "patent_abstract", "patent_date",
                                     "assignees.assignee_organization"]),
                    "o": json.dumps({"per_page": 10})
                }
            )
            if resp.status_code == 200:
                return resp.json().get("patents", []) or []
    except Exception as e:
        print(f"USPTO exception: {e}")
    return []


def normalize_lens(p: dict) -> dict:
    biblio = p.get("biblio", {}) or {}

    # Title: try biblio.invention_title first, then top-level title
    title = ""
    for titles in [biblio.get("invention_title", []), p.get("title", [])]:
        if isinstance(titles, list) and titles:
            title = titles[0].get("text", "") if isinstance(titles[0], dict) else str(titles[0])
            if title:
                break
    if not title:
        title = p.get("title", "未知标题") if isinstance(p.get("title"), str) else "未知标题"

    # Abstract
    abstract_list = p.get("abstract", []) or []
    abstract = (abstract_list[0].get("text", "") if abstract_list else "")[:500]

    # Applicants
    parties = biblio.get("parties", {}) or {}
    applicants = parties.get("applicants", []) or []
    names = []
    for a in applicants[:2]:
        name = (a.get("extracted_name") or {}).get("value") or a.get("name", "")
        if name:
            names.append(name)
    applicant = "、".join(names) or "未知申请人"

    jurisdiction = p.get("jurisdiction", "")
    source_map = {"CN": "中国专利", "US": "美国专利", "EP": "欧洲专利", "WO": "PCT国际专利"}

    return {
        "title": title,
        "abstract": abstract,
        "applicant": applicant,
        "date": p.get("publication_date", p.get("date_published", "")),
        "jurisdiction": jurisdiction,
        "source": source_map.get(jurisdiction, f"{jurisdiction}专利" if jurisdiction else "国际专利")
    }


def normalize_uspto(p: dict) -> dict:
    assignees = p.get("assignees", []) or []
    names = [a.get("assignee_organization", "") for a in assignees[:2] if a.get("assignee_organization")]
    return {
        "title": p.get("patent_title", ""),
        "abstract": (p.get("patent_abstract") or "")[:500],
        "applicant": "、".join(names) or "未知申请人",
        "date": p.get("patent_date", ""),
        "jurisdiction": "US",
        "source": "美国专利"
    }


MOCKUP_JSON_SPEC = (
    "根据上述产品方向，生成3个手机屏幕的界面内容数据，用于展示操作流程。\n\n"
    "只返回如下格式的JSON，不要其他任何文字：\n"
    '{\n'
    '  "screens": [\n'
    '    {\n'
    '      "label": "STEP 01 · 发现入口",\n'
    '      "title": "导航栏标题（5字以内）",\n'
    '      "color": "#1677FF",\n'
    '      "rows": [\n'
    '        {"type":"card","title":"具体内容标题","subtitle":"辅助描述","tags":["标签"]},\n'
    '        {"type":"list","icon":"📍","label":"功能项名称","value":"右侧数值","badge":false},\n'
    '        {"type":"divider"},\n'
    '        {"type":"section","text":"分组名"},\n'
    '        {"type":"chat_out","text":"用户发送的消息"},\n'
    '        {"type":"chat_in","text":"系统或对方的回复"},\n'
    '        {"type":"tag_row","tags":["标签A","标签B","标签C"]}\n'
    '      ],\n'
    '      "button": "主按钮文字（有tabbar时省略此字段）",\n'
    '      "tabbar": [\n'
    '        {"icon":"🏠","label":"首页","active":true},\n'
    '        {"icon":"💬","label":"消息","active":false},\n'
    '        {"icon":"👤","label":"我的","active":false}\n'
    '      ]\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "规则：\n"
    "1. screens 固定3项，分别展示：发现入口 / 核心操作 / 结果反馈\n"
    "2. 每屏 rows 最多8条，内容必须与产品方向强相关，使用真实具体的数值/名称，禁止占位符\n"
    "3. button 和 tabbar 二选一（有tabbar不要button字段）\n"
    "4. color 根据产品类型：科技/工具→#1677FF，创意→#722ED1，社交→#FA8C16，效率→#52C41A，金融→#096DD9\n"
    "5. label 格式固定：'STEP 0X · 场景名'，场景名4字以内\n"
    "6. badge:true 时 value 以主题色高亮显示（用于未读数、提醒等）\n"
)


async def analyze_results(idea: str, cn: list, intl: list) -> dict:
    def fmt(patents, label):
        if not patents:
            return f"【{label}】未找到相关专利\n"
        out = f"【{label}】共 {len(patents)} 项，取前几项：\n"
        for i, p in enumerate(patents[:5], 1):
            out += (
                f"{i}. 标题：{p['title']}\n"
                f"   申请人：{p['applicant']}  日期：{p['date']}\n"
                f"   摘要：{p['abstract'][:200]}\n\n"
            )
        return out

    prompt = (
        f'用户的想法是："{idea}"\n\n'
        f"专利检索结果：\n{fmt(cn, '中国专利')}\n{fmt(intl, '国际专利（美国/欧洲）')}\n\n"
        "根据以上结果，用通俗易懂的中文给出分析报告。不要堆砌专利术语。\n\n"
        "如果找到相似专利：选最相似的（最多5项），每项用2-3句话说清楚它是做什么的。\n"
        "如果没找到：评估创新性（1-10分），给出恰好3条差异化产品方向建议，3条方向必须覆盖不同用户群/场景/商业模式。\n\n"
        "只返回JSON，不要其他文字：\n"
        '{\n'
        '  "status": "found" 或 "novel",\n'
        '  "similar_patents": [\n'
        '    {"rank":1,"source":"中国专利/美国专利/欧洲专利","original_title":"原始标题",\n'
        '     "plain_title":"通俗标题","plain_description":"2-3句话","applicant":"申请人",\n'
        '     "date":"日期","similarity":"高度相似/中度相似/低度相似","key_difference":"1句话"}\n'
        '  ],\n'
        '  "novel_analysis": {\n'
        '    "innovation_score": 7,\n'
        '    "assessment": "对创新性的整体评价（3-4句话）",\n'
        '    "suggestions": [\n'
        '      {"text": "方向A：具体建议，说清楚差异化价值（2-3句话）"},\n'
        '      {"text": "方向B：与A明显不同的产品角度（2-3句话）"},\n'
        '      {"text": "方向C：第三个差异化方向（2-3句话）"}\n'
        '    ],\n'
        '    "patent_tips": "落地建议（2-3句话）"\n'
        '  },\n'
        '  "summary": "给用户的总结（2-4句话）"\n'
        '}\n'
    )

    resp = await deepseek.chat.completions.create(
        model="deepseek-chat",
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_json_safe(resp.choices[0].message.content)


async def generate_mockup(idea: str, suggestion_text: str, direction: str) -> dict:
    prompt = (
        f'用户的产品想法："{idea}"\n'
        f'当前产品方向（{direction}）：{suggestion_text}\n\n'
        f"{MOCKUP_JSON_SPEC}"
    )
    resp = await deepseek.chat.completions.create(
        model="deepseek-chat",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return parse_json_safe(resp.choices[0].message.content)
    except Exception:
        return {"screens": []}


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze(req: IdeaRequest):
    async def stream():
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            yield sse({"step": "keywords", "message": "正在理解你的想法，提取检索关键词..."})
            keywords = await extract_keywords(req.idea)
            yield sse({"step": "keywords_done", "keywords": keywords})

            yield sse({"step": "cn_search", "message": "正在检索中国专利数据库..."})
            cn_query = " ".join(keywords.get("zh_keywords", [])) + " " + keywords.get("search_query", "")
            cn_raw = await search_lens(cn_query.strip(), "CN", size=10)
            cn_patents = [normalize_lens(p) for p in cn_raw]
            yield sse({"step": "cn_done", "count": len(cn_patents)})

            yield sse({"step": "intl_search", "message": "正在检索美国、欧洲专利数据库..."})
            en_query = keywords.get("search_query", " ".join(keywords.get("en_keywords", [])))
            us_raw, ep_raw, uspto_raw = await asyncio.gather(
                search_lens(en_query, "US", size=8),
                search_lens(en_query, "EP", size=5),
                search_uspto(en_query)
            )
            intl_patents = [normalize_lens(p) for p in us_raw + ep_raw]
            if not intl_patents:
                intl_patents = [normalize_uspto(p) for p in uspto_raw]
            yield sse({"step": "intl_done", "count": len(intl_patents)})

            yield sse({"step": "analyzing", "message": "AI 正在分析专利关联性，生成建议方向..."})
            report = await analyze_results(req.idea, cn_patents, intl_patents)
            report["cn_count"] = len(cn_patents)
            report["intl_count"] = len(intl_patents)
            report["keywords"] = keywords

            n = report.get("novel_analysis") or {}
            sugs = n.get("suggestions") or []
            # 保底：确保始终有 3 个方向
            while len(sugs) < 3:
                sugs.append({"text": f"方向{'ABC'[len(sugs)]}：AI 生成内容暂缺，请重新搜索。"})
            n["suggestions"] = sugs
            if report.get("status") == "novel":
                labels = ["A", "B", "C", "D", "E"]
                yield sse({"step": "mockups", "message": f"正在生成 {len(sugs)} 个方案示意图..."})
                mockup_htmls = await asyncio.gather(*[
                    generate_mockup(req.idea, (s.get("text") or s) if isinstance(s, dict) else s, labels[i])
                    for i, s in enumerate(sugs)
                ])
                for i, data in enumerate(mockup_htmls):
                    sugs[i]["mockup_data"] = data

            yield sse({"step": "done", "data": report})

        except Exception as e:
            yield sse({"step": "error", "message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    if port == 8000:
        import threading, webbrowser, time
        def open_browser():
            time.sleep(1.5)
            webbrowser.open("http://localhost:8000")
        threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
