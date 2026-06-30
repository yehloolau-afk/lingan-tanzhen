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


MOCKUP_SPEC = (
    "生成一个完整的产品方案可视化 HTML，包含两部分：上方是「产品操作流程图」，下方是「核心界面示意」。\n\n"

    "═══ 第一部分：产品操作流程图 ═══\n"
    "展示用户使用该产品的完整操作路径，4-6个步骤节点横向排列，节点间用箭头连接。\n\n"
    "整体容器：width:100%; box-sizing:border-box; background:#FFF; padding:28px 24px 20px; border-bottom:1px solid #F0F0F0;\n"
    "标题行：font-size:11px; font-weight:700; color:#8C8C8C; letter-spacing:1px; text-transform:uppercase;\n"
    "  text-align:center; margin-bottom:16px; 内容：PRODUCT FLOW\n\n"
    "流程行：display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:nowrap;\n\n"
    "每个节点（step-node）：\n"
    "  外层：display:flex; flex-direction:column; align-items:center; gap:6px;\n"
    "  图标圆圈：width:44px; height:44px; border-radius:50%; background:主色浅15%色;\n"
    "    display:flex; align-items:center; justify-content:center; font-size:20px;\n"
    "  步骤序号：font-size:9px; font-weight:700; color:主色; 叠在图标右上角或单独显示\n"
    "  节点标签：font-size:11px; font-weight:600; color:#1A1A1A; text-align:center; max-width:60px;\n"
    "  副标签：font-size:10px; color:#8C8C8C; text-align:center; max-width:60px;\n\n"
    "节点之间的箭头：\n"
    "  width:28px; height:2px; background:linear-gradient(90deg,主色 0%,主色浅50% 100%);\n"
    "  position:relative; flex-shrink:0;\n"
    "  末端三角箭头：用::after伪元素 border-left:6px solid 主色; border-top:4px solid transparent; border-bottom:4px solid transparent;\n\n"
    "节点类型区分（用不同图标和圆圈颜色）：\n"
    "  触发节点（第1个）：圆圈background主色; emoji白色; 代表用户进入\n"
    "  操作节点（中间）：圆圈background主色浅15%色; emoji主色\n"
    "  结果节点（最后）：圆圈background:#52C41A浅15%色; emoji绿色; 代表完成\n\n"
    "决策分支（可选，如有关键分叉）：在相关节点下方加小的分支标签（成功/失败、已有/新建等）\n\n"

    "═══ 第二部分：核心界面示意 ═══\n"
    "3个手机屏幕横向排列（可侧滑浏览操作流程），严格遵循 iPhone 屏幕比例（9:19.5），展示产品核心操作界面。\n\n"
    "【重要】整体容器必须自带横向滚动，不依赖外层容器：\n"
    "整体容器：width:100%; box-sizing:border-box; background:#F5F5F7; padding:28px 0 40px; overflow-x:auto; -webkit-overflow-scrolling:touch;\n"
    "标题行：font-size:11px; font-weight:700; color:#8C8C8C; letter-spacing:1px; text-transform:uppercase;\n"
    "  text-align:center; margin-bottom:20px; padding:0 24px; 内容：KEY SCREENS\n"
    "屏幕行：display:flex; align-items:flex-start; gap:24px; width:max-content; min-width:100%; padding:0 24px; box-sizing:border-box;\n\n"
    "每个手机单元（步骤标签 + 手机外框）：\n"
    "步骤标签：text-align:center; font-size:11px; font-weight:700; color:#8E8E93; letter-spacing:0.5px; margin-bottom:10px;\n"
    "手机外框：position:relative; width:300px; flex-shrink:0; border-radius:10px; overflow:hidden;\n"
    "  border:2px solid #1A1A1A; background:#FFF; box-sizing:border-box;\n\n"
    "屏幕内容区（严格比例，9:19.5）：\n"
    "  background:#F2F2F7; width:100%; box-sizing:border-box;\n"
    "  height:641px（严格固定，不可改变）\n"
    "  overflow:hidden; display:flex; flex-direction:column; position:relative;\n\n"
    "每屏层次（高度分配，总计641px）：\n"
    "① 导航栏：height:52px; flex-shrink:0; padding:0 20px; display:flex; align-items:center; justify-content:space-between;\n"
    "   background:#FFF; border-bottom:0.5px solid #F0F0F0;\n"
    "   中间标题font-size:14px font-weight:700 color:#000; 左侧'‹ 返回' font-size:13px color:主色\n"
    "② 内容区：flex:1（自动填充剩余高度）；overflow:hidden\n"
    "   卡片间距padding:14px 18px; 卡片font-size:15px; 列表标题font-size:15px; 辅助文字font-size:13px\n"
    "③ 底部主按钮（若有）：height:54px; flex-shrink:0; margin:0 18px 16px; border-radius:14px; font-size:17px font-weight:600\n"
    "   或 Tab Bar：height:54px; flex-shrink:0; border-top:0.5px solid #F0F0F0; font-size:11px\n\n"
    "连接箭头（2个）：align-self:center; padding-bottom:60px; font-size:24px; color:#C7C7CC; flex-shrink:0;\n\n"

    "═══ Ant Design Mobile 组件规范（屏幕宽度296px，严格执行） ═══\n"
    "主色：科技/工具→#1677FF；创意→#722ED1；社交→#FA8C16；效率→#52C41A；金融→#096DD9\n\n"
    "卡片：background:#FFF; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:8px 12px; margin:0 10px 7px;\n"
    "列表行：min-height:38px; padding:0 14px; display:flex; align-items:center; background:#FFF;\n"
    "  border-bottom:1px solid #F5F5F5; 文字font-size:12px color:#000; 右侧chevron › color:#C7C7CC font-size:11px\n"
    "  列表组之间用：height:8px; background:#F5F5F5; 作为分组间隔\n"
    "Tag/徽标：display:inline-block; background:rgba(主色,0.1); color:主色;\n"
    "  border-radius:4px; padding:2px 6px; font-size:10px; font-weight:500;\n"
    "  【禁止】border-style:dashed 或任何虚线/描边边框\n"
    "主按钮：display:block; background:主色; color:#FFF; border-radius:10px;\n"
    "  height:40px; line-height:40px; font-size:13px; font-weight:600; text-align:center;\n"
    "  margin:0 14px; letter-spacing:0.5px;\n"
    "次按钮：border:1.5px solid 主色; color:主色; background:#FFF; border-radius:10px;\n"
    "  height:36px; line-height:34px; font-size:12px; text-align:center; margin:0 14px;\n"
    "输入框：background:#F5F5F5; border-radius:8px; border:none; padding:8px 12px; font-size:12px; color:#000;\n"
    "文字层次：大标题16px/700 color:#000; 正文12px/400 color:#000; 辅助10px/400 color:#8C8C8C\n"
    "Tab Bar：height:44px; flex-shrink:0; background:#FFF; border-top:0.5px solid #F0F0F0;\n"
    "  display:flex; 每项：flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;\n"
    "  emoji font-size:16px; 文字font-size:9px; active color:主色; inactive color:#8C8C8C\n\n"

    "═══ 严格禁止（这些会让界面看起来像线框图） ═══\n"
    "✗ 任何虚线边框：border:dashed、border-style:dashed、outline:dashed\n"
    "✗ 手势/光标 emoji 作为 UI 元素：👆👋✋☝️🖱️🖊️\n"
    "✗ 界面内的说明性文字：「点击此处」「长按选择」「滑动查看」「此处放xxx」\n"
    "✗ 半透明占位色块代替真实内容\n"
    "✗ 线框/描边风格的图标（用实心 emoji 替代）\n"
    "✗ 浮动在屏幕边缘的提示箭头或标注\n"
    "✗ 超过2层 z-index 叠层（保持扁平布局）\n"
    "✗ 在流程图区块/section容器上加任何彩色或有色描边边框（红/橙/蓝等），容器只用背景色区分\n"
    "✗ 任何内容超出手机外框边界：所有子元素必须在 overflow:hidden 的外框内，禁止 position:fixed、禁止负margin溢出\n"
    "✗ 将 HTML 源码作为文本内容渲染（如 background-size:cover> 等代码出现在屏幕上）\n\n"

    "═══ 内容要求 ═══\n"
    "所有文字（节点名、屏幕标题、列表项、按钮文字）必须与「用户想法」和「该建议方向」强相关\n"
    "使用真实产品名/功能名/数值，禁止「示例」「占位符」「xxx」「产品名」\n"
    "每屏必须有实质内容：至少2个列表项或1张卡片 + 1个操作按钮，视觉上像已发布的产品\n"
    "3屏分别展示：发现入口（首屏有数据）/ 核心操作（功能界面）/ 结果反馈（完成状态）\n\n"

    "═══ 代码要求 ═══\n"
    "HTML根结构必须是：<html><head><meta charset='UTF-8'></head><body style='margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,PingFang SC,sans-serif'>...内容...</body></html>\n"
    "屏幕内容区必须设置：width:224px; height:486px; overflow:hidden; display:flex; flex-direction:column;\n"
    "内容绝对不能溢出屏幕边界，每屏内容控制在 486px 高度内完整展示\n"
    "只返回完整可渲染的 HTML，不要markdown代码块，不要HTML注释，不要多余空行，总长度控制在5500字以内。"
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


async def generate_mockup(idea: str, suggestion_text: str, direction: str) -> str:
    prompt = (
        f'用户的产品想法："{idea}"\n'
        f'当前产品方向（{direction}）：{suggestion_text}\n\n'
        f"根据以上方向，{MOCKUP_SPEC}"
    )
    resp = await deepseek.chat.completions.create(
        model="deepseek-chat",
        max_tokens=5500,
        messages=[{"role": "user", "content": prompt}]
    )
    html = resp.choices[0].message.content.strip()
    if html.startswith("```"):
        html = html.split("```")[1].lstrip("html").strip()
    return html


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
                for i, html in enumerate(mockup_htmls):
                    sugs[i]["mockup_html"] = html

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
