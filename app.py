import os
import json
import time
import threading

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import anthropic

load_dotenv()

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN", "vergesense")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
DATA_URL = os.getenv(
    "DATA_URL",
    "https://github.com/joshuaguyervs/vs-kb-analyzer/releases/download/1.0/tickets.json"
)




zendesk_auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
state = {
    "tickets": [],
    "kb_articles": [],
    "loading": False,
    "load_status": "idle",  # idle | loading | ready | error
    "load_message": "",
    "analysis_cache": {}
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tickets():
    """Download and parse the ticket export from GitHub Releases."""
    state["loading"] = True
    state["load_status"] = "loading"
    state["load_message"] = "Downloading ticket data..."

    try:
        state["load_message"] = f"Downloading ticket export (~330MB, this takes a minute)..."
        resp = requests.get(DATA_URL, timeout=300, stream=True)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code} fetching data file")
        state["load_message"] = "Parsing ticket data..."
        data = json.loads(resp.content)
        state["tickets"] = data.get("tickets", [])
        state["load_message"] = f"Loaded {len(state['tickets'])} tickets."
    except Exception as e:
        state["load_status"] = "error"
        state["load_message"] = f"Data load error: {str(e)}"
        state["loading"] = False
        return

    # Now load KB articles
    state["load_message"] = "Fetching Zendesk KB articles..."
    try:
        articles = fetch_all_kb_articles()
        state["kb_articles"] = articles
        state["load_message"] = f"Ready. {len(state['tickets'])} tickets · {len(state['kb_articles'])} KB articles loaded."
        state["load_status"] = "ready"
    except Exception as e:
        state["load_status"] = "error"
        state["load_message"] = f"Zendesk KB error: {str(e)}"

    state["loading"] = False


def fetch_all_kb_articles():
    """Fetch all articles from Zendesk Help Center."""
    articles = []
    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/help_center/articles.json?per_page=100"
    while url:
        resp = requests.get(url, auth=zendesk_auth, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Zendesk API {resp.status_code}: {resp.text}")
        data = resp.json()
        articles.extend(data.get("articles", []))
        url = data.get("next_page")
    return articles


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def ticket_summary_text(ticket):
    """Flatten a ticket into a single readable block for Claude."""
    lines = [
        f"Subject: {ticket.get('subject', '')}",
        f"Status: {ticket.get('status', '')}",
        f"Tags: {', '.join(ticket.get('tags', []))}",
    ]
    comments = ticket.get("comments", [])
    public = [c for c in comments if c.get("public")]
    for c in public[:6]:  # cap at 6 comments to stay within token limits
        role = "Customer" if c.get("author_id") else "Agent"
        lines.append(f"{role}: {c.get('body', '')[:500]}")
    return "\n".join(lines)


def call_claude(system, user, max_tokens=2000):
    """Single Claude call, returns text."""
    msg = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return msg.content[0].text


def call_claude_json(system, user, max_tokens=2000):
    """Claude call that returns parsed JSON."""
    system += "\n\nRespond ONLY with valid JSON. No markdown, no backticks, no explanation."
    text = call_claude(system, user, max_tokens)
    # Strip any accidental markdown fences
    text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def analyze_topic_clusters(tickets, kb_articles):
    """
    Cluster tickets into topics, count frequency, and identify KB gaps.
    Returns list of cluster dicts.
    """
    cache_key = "clusters"
    if cache_key in state["analysis_cache"]:
        return state["analysis_cache"][cache_key]

    # Sample up to 300 tickets for clustering (representative sample)
    sample = tickets[:300]
    ticket_texts = []
    for t in sample:
        ticket_texts.append(f"- [{t.get('id')}] {t.get('subject', '')} | Tags: {', '.join(t.get('tags', []))}")

    kb_titles = [a.get("title", "") for a in kb_articles]

    system = """You are a support operations analyst. Your job is to identify the main topic clusters 
from a list of support tickets and determine which ones have adequate KB coverage."""

    user = f"""Here are {len(sample)} support ticket subjects and tags:

{chr(10).join(ticket_texts)}

Current KB article titles:
{chr(10).join(f'- {t}' for t in kb_titles)}

Identify 10-15 distinct topic clusters from the tickets. For each cluster return:
- id: short snake_case identifier
- name: human-readable cluster name
- description: one sentence describing what issues this covers
- ticket_count_estimate: estimated number of tickets in this cluster (as integer)
- has_kb_coverage: true/false whether an existing KB article adequately covers this
- kb_article_title: if has_kb_coverage is true, the matching article title, else null
- gap_severity: "high" | "medium" | "low" (how badly a new article is needed)
- representative_subjects: array of 3 example ticket subjects from the list

Return a JSON array of cluster objects."""

    result = call_claude_json(system, user, max_tokens=3000)
    state["analysis_cache"][cache_key] = result
    return result


def analyze_high_volume(tickets):
    """Return tag frequency analysis."""
    cache_key = "tag_freq"
    if cache_key in state["analysis_cache"]:
        return state["analysis_cache"][cache_key]

    tag_counts = {}
    for t in tickets:
        for tag in t.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Sort and take top 30
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    result = [{"tag": t, "count": c} for t, c in sorted_tags]
    state["analysis_cache"][cache_key] = result
    return result


def analyze_article_improvement(article):
    """Given a KB article, find related tickets and suggest improvements."""
    article_id = article.get("id")
    cache_key = f"improve_{article_id}"
    if cache_key in state["analysis_cache"]:
        return state["analysis_cache"][cache_key]

    title = article.get("title", "")
    body = article.get("body", "")[:3000]  # cap body

    # Find tickets that likely relate to this article by title keyword match
    keywords = [w.lower() for w in title.split() if len(w) > 3]
    related = []
    for t in state["tickets"]:
        subj = t.get("subject", "").lower()
        if any(k in subj for k in keywords):
            related.append(t)
        if len(related) >= 20:
            break

    related_texts = "\n\n---\n\n".join(ticket_summary_text(t) for t in related[:10])

    system = """You are a technical writer and support operations expert. 
Analyze a KB article against real customer support tickets to identify improvements."""

    user = f"""KB Article: "{title}"

Article content (truncated):
{body}

Related support tickets that may indicate gaps in this article:
{related_texts if related_texts else "No closely matching tickets found."}

Based on this analysis, provide:
1. A quality score for the article (0-100)
2. Key issues found (missing steps, outdated info, unclear sections)
3. Specific improvement suggestions
4. A list of common customer questions this article doesn't answer

Return as JSON with keys:
- quality_score: integer 0-100
- issues: array of strings
- suggestions: array of strings  
- unanswered_questions: array of strings
- ticket_count: integer (how many related tickets you found)
- summary: one sentence overall assessment"""

    result = call_claude_json(system, user, max_tokens=2000)
    result["article_id"] = article_id
    result["article_title"] = title
    result["related_ticket_count"] = len(related)
    state["analysis_cache"][cache_key] = result
    return result


def analyze_outdated_articles(articles, tickets):
    """Flag articles that tickets suggest are outdated or wrong."""
    cache_key = "outdated"
    if cache_key in state["analysis_cache"]:
        return state["analysis_cache"][cache_key]

    # Look for tickets where customers say docs were wrong/outdated
    complaint_keywords = [
        "documentation", "docs", "article", "guide", "instructions",
        "outdated", "wrong", "incorrect", "doesn't work", "not working",
        "steps don't", "followed the", "as per the"
    ]

    flagged_tickets = []
    for t in tickets:
        comments = t.get("comments", [])
        for c in comments:
            body = c.get("body", "").lower()
            if any(k in body for k in complaint_keywords):
                flagged_tickets.append({
                    "id": t.get("id"),
                    "subject": t.get("subject"),
                    "snippet": c.get("body", "")[:300]
                })
                break
        if len(flagged_tickets) >= 50:
            break

    if not flagged_tickets:
        result = []
        state["analysis_cache"][cache_key] = result
        return result

    kb_titles = [{"id": a.get("id"), "title": a.get("title")} for a in articles]
    ticket_snippets = "\n\n".join(
        f"Ticket {t['id']} - {t['subject']}:\n{t['snippet']}"
        for t in flagged_tickets[:20]
    )

    system = """You are a support analyst identifying which KB articles may be outdated based on customer complaints."""

    user = f"""These support tickets contain customer complaints that suggest documentation issues:

{ticket_snippets}

Current KB articles:
{json.dumps(kb_titles, indent=2)}

Identify which KB articles are likely outdated or problematic based on the ticket evidence.
Return a JSON array where each item has:
- article_id: integer (match from the KB list, or null if no match)
- article_title: string
- reason: why this article may be outdated
- evidence_ticket_ids: array of ticket IDs that suggest this
- confidence: "high" | "medium" | "low"

Only include articles where there is real evidence from the tickets."""

    result = call_claude_json(system, user, max_tokens=2000)
    state["analysis_cache"][cache_key] = result
    return result


def generate_new_article_draft(cluster):
    """Generate a draft KB article for a gap cluster."""
    cluster_name = cluster.get("name", "")
    cluster_desc = cluster.get("description", "")
    cache_key = f"draft_{cluster.get('id', cluster_name)}"
    if cache_key in state["analysis_cache"]:
        return state["analysis_cache"][cache_key]

    # Find representative tickets for this cluster
    keywords = [w.lower() for w in cluster_name.split() if len(w) > 3]
    related = []
    for t in state["tickets"]:
        subj = t.get("subject", "").lower()
        if any(k in subj for k in keywords):
            related.append(t)
        if len(related) >= 10:
            break

    ticket_context = "\n\n---\n\n".join(ticket_summary_text(t) for t in related[:5])

    system = """You are a technical writer creating Zendesk Help Center articles for VergeSense, 
a workplace occupancy intelligence platform. Write clear, concise, customer-facing documentation."""

    user = f"""Write a Zendesk KB article for the topic: "{cluster_name}"

Topic description: {cluster_desc}

Here are example support tickets on this topic to guide the content:
{ticket_context if ticket_context else "No example tickets available."}

Write a complete KB article with:
- A clear, searchable title
- A brief intro (1-2 sentences)  
- Step-by-step instructions or explanation
- A troubleshooting section if applicable
- Clear, friendly tone suitable for IT admins and facilities managers

Return as JSON with keys:
- title: string
- intro: string
- sections: array of objects with "heading" and "content" keys
- troubleshooting: array of objects with "problem" and "solution" keys"""

    result = call_claude_json(system, user, max_tokens=3000)
    state["analysis_cache"][cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML

HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>KB Intelligence · VergeSense</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">\n<style>\n  :root {\n    --bg: #0b0f1a;\n    --surface: #111827;\n    --surface2: #1a2235;\n    --border: #1f2d45;\n    --border2: #2a3d5a;\n    --text: #e2e8f0;\n    --muted: #64748b;\n    --amber: #f59e0b;\n    --amber-dim: rgba(245,158,11,0.12);\n    --green: #10b981;\n    --green-dim: rgba(16,185,129,0.1);\n    --red: #ef4444;\n    --red-dim: rgba(239,68,68,0.1);\n    --blue: #3b82f6;\n    --blue-dim: rgba(59,130,246,0.1);\n    --mono: \'IBM Plex Mono\', monospace;\n    --sans: \'IBM Plex Sans\', sans-serif;\n  }\n  * { box-sizing: border-box; margin: 0; padding: 0; }\n  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; min-height: 100vh; }\n\n  .shell { display: grid; grid-template-columns: 220px 1fr; grid-template-rows: 56px 1fr; min-height: 100vh; }\n\n  .topbar { grid-column: 1 / -1; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 24px; gap: 16px; }\n  .topbar-logo { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--amber); letter-spacing: 0.05em; text-transform: uppercase; }\n  .topbar-sep { color: var(--border2); font-size: 18px; }\n  .topbar-title { color: var(--muted); font-size: 13px; }\n  .topbar-status { margin-left: auto; font-family: var(--mono); font-size: 11px; padding: 4px 10px; border-radius: 3px; border: 1px solid var(--border2); color: var(--muted); display: flex; align-items: center; gap: 6px; }\n  .topbar-status.ready { color: var(--green); border-color: var(--green); }\n  .topbar-status.loading { color: var(--amber); border-color: var(--amber); }\n  .topbar-status.error { color: var(--red); border-color: var(--red); }\n  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }\n  .status-dot.pulse { animation: pulse 1.4s ease-in-out infinite; }\n  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }\n\n  .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 20px 0; display: flex; flex-direction: column; gap: 2px; }\n  .nav-section { padding: 8px 16px 4px; font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }\n  .nav-item { display: flex; align-items: center; gap: 10px; padding: 8px 16px; cursor: pointer; color: var(--muted); font-size: 13px; font-weight: 500; transition: all 0.15s; border-left: 2px solid transparent; }\n  .nav-item:hover { color: var(--text); background: var(--surface2); }\n  .nav-item.active { color: var(--amber); border-left-color: var(--amber); background: var(--amber-dim); }\n  .nav-icon { font-size: 15px; width: 18px; text-align: center; }\n  .nav-badge { margin-left: auto; font-family: var(--mono); font-size: 10px; background: var(--surface2); padding: 1px 6px; border-radius: 10px; color: var(--muted); }\n  .nav-item.active .nav-badge { background: var(--amber-dim); color: var(--amber); }\n\n  .main { overflow-y: auto; padding: 28px 32px; display: flex; flex-direction: column; gap: 24px; }\n\n  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }\n  .panel-header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }\n  .panel-title { font-family: var(--mono); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text); }\n  .panel-subtitle { font-size: 12px; color: var(--muted); margin-left: auto; }\n  .panel-body { padding: 20px; }\n\n  .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }\n  .stat { background: var(--surface); padding: 20px 24px; display: flex; flex-direction: column; gap: 4px; }\n  .stat-value { font-family: var(--mono); font-size: 28px; font-weight: 600; color: var(--text); line-height: 1; }\n  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }\n  .stat-value.amber { color: var(--amber); }\n  .stat-value.green { color: var(--green); }\n  .stat-value.red { color: var(--red); }\n\n  .data-table { width: 100%; border-collapse: collapse; }\n  .data-table th { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 500; }\n  .data-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; vertical-align: top; }\n  .data-table tr:last-child td { border-bottom: none; }\n  .data-table tbody tr:hover { background: var(--surface2); }\n\n  .badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 3px; font-family: var(--mono); font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }\n  .badge-high { background: var(--red-dim); color: var(--red); border: 1px solid var(--red); }\n  .badge-medium { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); }\n  .badge-low { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }\n  .badge-gap { background: var(--red-dim); color: var(--red); }\n  .badge-covered { background: var(--green-dim); color: var(--green); }\n\n  .btn { display: inline-flex; align-items: center; gap: 6px; padding: 7px 14px; border-radius: 4px; font-family: var(--mono); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; cursor: pointer; border: 1px solid var(--border2); background: transparent; color: var(--muted); transition: all 0.15s; }\n  .btn:hover { color: var(--text); border-color: var(--text); }\n  .btn-primary { background: var(--amber); color: #000; border-color: var(--amber); }\n  .btn-primary:hover { background: #fbbf24; border-color: #fbbf24; color: #000; }\n  .btn:disabled { opacity: 0.4; cursor: not-allowed; }\n\n  .score-ring { display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; font-family: var(--mono); font-size: 14px; font-weight: 700; border: 2px solid; }\n  .score-high { color: var(--green); border-color: var(--green); }\n  .score-mid { color: var(--amber); border-color: var(--amber); }\n  .score-low { color: var(--red); border-color: var(--red); }\n\n  .draft-section { margin-bottom: 20px; }\n  .draft-section h3 { font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--amber); margin-bottom: 8px; }\n  .draft-section p { color: var(--text); line-height: 1.7; }\n\n  .tag-bar { display: flex; align-items: center; gap: 12px; padding: 5px 0; }\n  .tag-name { font-family: var(--mono); font-size: 11px; color: var(--muted); min-width: 160px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }\n  .tag-track { flex: 1; height: 4px; background: var(--surface2); border-radius: 2px; overflow: hidden; }\n  .tag-fill { height: 100%; background: var(--amber); border-radius: 2px; transition: width 0.6s ease; }\n  .tag-count { font-family: var(--mono); font-size: 11px; color: var(--muted); min-width: 40px; text-align: right; }\n\n  .page { display: none; }\n  .page.active { display: flex; flex-direction: column; gap: 24px; }\n\n  .empty { text-align: center; padding: 48px 24px; color: var(--muted); font-size: 13px; display: flex; flex-direction: column; align-items: center; gap: 12px; }\n  .empty-icon { font-size: 32px; }\n\n  .analyzing { display: flex; align-items: center; gap: 10px; padding: 16px 20px; color: var(--amber); font-family: var(--mono); font-size: 12px; }\n  .loading-spinner { width: 18px; height: 18px; border: 2px solid var(--border2); border-top-color: var(--amber); border-radius: 50%; animation: spin 0.8s linear infinite; flex-shrink: 0; }\n  @keyframes spin { to { transform: rotate(360deg); } }\n\n  .detail-panel { position: fixed; top: 56px; right: 0; width: 560px; height: calc(100vh - 56px); background: var(--surface); border-left: 1px solid var(--border); overflow-y: auto; z-index: 100; transform: translateX(100%); transition: transform 0.25s ease; }\n  .detail-panel.open { transform: translateX(0); }\n  .detail-header { position: sticky; top: 0; background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 20px; display: flex; align-items: center; gap: 12px; z-index: 1; }\n  .detail-close { margin-left: auto; background: none; border: none; color: var(--muted); cursor: pointer; font-size: 18px; padding: 4px; }\n  .detail-close:hover { color: var(--text); }\n  .detail-body { padding: 20px; display: flex; flex-direction: column; gap: 20px; }\n\n  .list-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px; color: var(--text); }\n  .list-item:last-child { border-bottom: none; }\n  .list-bullet { color: var(--amber); font-size: 16px; line-height: 1.4; flex-shrink: 0; }\n</style>\n</head>\n<body>\n<div class="shell">\n\n  <header class="topbar">\n    <span class="topbar-logo">VergeSense</span>\n    <span class="topbar-sep">/</span>\n    <span class="topbar-title">KB Intelligence</span>\n    <div class="topbar-status" id="topStatus">\n      <div class="status-dot pulse"></div>\n      <span id="topStatusText">Initializing...</span>\n    </div>\n  </header>\n\n  <nav class="sidebar">\n    <div class="nav-section">Analysis</div>\n    <div class="nav-item active" onclick="navTo(\'overview\', this)">\n      <span class="nav-icon">◈</span> Overview\n    </div>\n    <div class="nav-item" onclick="navTo(\'gaps\', this)">\n      <span class="nav-icon">◎</span> Gap Analysis\n      <span class="nav-badge" id="gapBadge">—</span>\n    </div>\n    <div class="nav-item" onclick="navTo(\'volume\', this)">\n      <span class="nav-icon">◷</span> High Volume\n    </div>\n    <div class="nav-section">Articles</div>\n    <div class="nav-item" onclick="navTo(\'articles\', this)">\n      <span class="nav-icon">◻</span> Existing Articles\n      <span class="nav-badge" id="articleBadge">—</span>\n    </div>\n    <div class="nav-item" onclick="navTo(\'outdated\', this)">\n      <span class="nav-icon">⚠</span> Outdated Flags\n    </div>\n  </nav>\n\n  <main class="main">\n\n    <div class="page active" id="page-overview">\n      <div class="stats-row">\n        <div class="stat"><div class="stat-value" id="statTickets">—</div><div class="stat-label">Total Tickets</div></div>\n        <div class="stat"><div class="stat-value green" id="statArticles">—</div><div class="stat-label">KB Articles</div></div>\n        <div class="stat"><div class="stat-value amber" id="statGaps">—</div><div class="stat-label">Coverage Gaps</div></div>\n        <div class="stat"><div class="stat-value red" id="statOutdated">—</div><div class="stat-label">Outdated Flags</div></div>\n      </div>\n      <div class="panel">\n        <div class="panel-header">\n          <span class="panel-title">Top Ticket Tags</span>\n          <span class="panel-subtitle">by frequency</span>\n        </div>\n        <div class="panel-body" id="tagBars">\n          <div class="analyzing"><div class="loading-spinner"></div>Loading...</div>\n        </div>\n      </div>\n    </div>\n\n    <div class="page" id="page-gaps">\n      <div class="panel">\n        <div class="panel-header">\n          <span class="panel-title">Topic Clusters · Gap Analysis</span>\n          <button class="btn btn-primary" onclick="runClusterAnalysis()" id="clusterBtn">Run Analysis</button>\n        </div>\n        <div id="clusterContent">\n          <div class="empty">\n            <div class="empty-icon">◎</div>\n            <div>Click "Run Analysis" to cluster tickets and identify KB gaps.</div>\n            <div style="font-size:11px; color:var(--muted)">Uses Claude to analyze up to 300 tickets. Takes ~30 seconds.</div>\n          </div>\n        </div>\n      </div>\n    </div>\n\n    <div class="page" id="page-volume">\n      <div class="panel">\n        <div class="panel-header">\n          <span class="panel-title">High Volume Topics</span>\n          <span class="panel-subtitle">top tags by ticket count</span>\n        </div>\n        <div class="panel-body" id="volumeContent">\n          <div class="analyzing"><div class="loading-spinner"></div>Loading...</div>\n        </div>\n      </div>\n    </div>\n\n    <div class="page" id="page-articles">\n      <div class="panel">\n        <div class="panel-header">\n          <span class="panel-title">Existing KB Articles</span>\n          <span class="panel-subtitle">click any row to run improvement analysis</span>\n        </div>\n        <div id="articlesContent">\n          <div class="analyzing"><div class="loading-spinner"></div>Loading articles...</div>\n        </div>\n      </div>\n    </div>\n\n    <div class="page" id="page-outdated">\n      <div class="panel">\n        <div class="panel-header">\n          <span class="panel-title">Outdated Article Flags</span>\n          <button class="btn btn-primary" onclick="runOutdatedAnalysis()" id="outdatedBtn">Run Analysis</button>\n        </div>\n        <div id="outdatedContent">\n          <div class="empty">\n            <div class="empty-icon">⚠</div>\n            <div>Click "Run Analysis" to scan tickets for documentation complaints.</div>\n          </div>\n        </div>\n      </div>\n    </div>\n\n  </main>\n</div>\n\n<div class="detail-panel" id="detailPanel">\n  <div class="detail-header">\n    <span class="panel-title" id="detailTitle">Detail</span>\n    <button class="detail-close" onclick="closeDetail()">✕</button>\n  </div>\n  <div class="detail-body" id="detailBody"></div>\n</div>\n\n<script>\nlet appReady = false;\nlet overviewLoaded = false;\nlet articlesLoaded = false;\nlet volumeLoaded = false;\n\nfunction poll() {\n  fetch(\'/api/status\').then(r => r.json()).then(d => {\n    const el = document.getElementById(\'topStatus\');\n    const txt = document.getElementById(\'topStatusText\');\n    el.className = \'topbar-status \' + d.status;\n    const dot = el.querySelector(\'.status-dot\');\n    if (d.status === \'loading\') {\n      dot.classList.add(\'pulse\');\n      txt.textContent = d.message;\n      setTimeout(poll, 2000);\n    } else if (d.status === \'ready\') {\n      dot.classList.remove(\'pulse\');\n      txt.textContent = d.ticket_count.toLocaleString() + \' tickets · \' + d.article_count + \' articles\';\n      document.getElementById(\'statTickets\').textContent = d.ticket_count.toLocaleString();\n      document.getElementById(\'statArticles\').textContent = d.article_count;\n      document.getElementById(\'articleBadge\').textContent = d.article_count;\n      appReady = true;\n      loadOverview();\n      loadArticles();\n      loadVolume();\n    } else if (d.status === \'error\') {\n      dot.classList.remove(\'pulse\');\n      txt.textContent = \'Error: \' + d.message;\n    } else {\n      setTimeout(poll, 1000);\n    }\n  }).catch(() => setTimeout(poll, 3000));\n}\n\nfunction navTo(page, el) {\n  document.querySelectorAll(\'.page\').forEach(p => p.classList.remove(\'active\'));\n  document.querySelectorAll(\'.nav-item\').forEach(n => n.classList.remove(\'active\'));\n  document.getElementById(\'page-\' + page).classList.add(\'active\');\n  el.classList.add(\'active\');\n  closeDetail();\n}\n\nfunction loadOverview() {\n  if (overviewLoaded) return;\n  fetch(\'/api/overview\').then(r => r.json()).then(d => {\n    overviewLoaded = true;\n    const max = d.tag_frequency[0]?.count || 1;\n    document.getElementById(\'tagBars\').innerHTML = d.tag_frequency.map(t =>\n      \'<div class="tag-bar">\' +\n        \'<div class="tag-name">\' + t.tag + \'</div>\' +\n        \'<div class="tag-track"><div class="tag-fill" style="width:\' + (t.count/max*100).toFixed(1) + \'%"></div></div>\' +\n        \'<div class="tag-count">\' + t.count + \'</div>\' +\n      \'</div>\'\n    ).join(\'\');\n  });\n}\n\nfunction loadVolume() {\n  if (volumeLoaded) return;\n  fetch(\'/api/overview\').then(r => r.json()).then(d => {\n    volumeLoaded = true;\n    const tags = d.tag_frequency;\n    const max = tags[0]?.count || 1;\n    document.getElementById(\'volumeContent\').innerHTML =\n      \'<table class="data-table"><thead><tr><th>Tag</th><th>Tickets</th><th>Distribution</th></tr></thead><tbody>\' +\n      tags.map(t =>\n        \'<tr><td><span style="font-family:var(--mono);font-size:12px;">\' + t.tag + \'</span></td>\' +\n        \'<td><span style="font-family:var(--mono);color:var(--amber);">\' + t.count + \'</span></td>\' +\n        \'<td style="min-width:200px;"><div class="tag-track" style="height:6px;"><div class="tag-fill" style="width:\' + (t.count/max*100).toFixed(1) + \'%"></div></div></td></tr>\'\n      ).join(\'\') + \'</tbody></table>\';\n  });\n}\n\nfunction runClusterAnalysis() {\n  if (!appReady) return;\n  document.getElementById(\'clusterBtn\').disabled = true;\n  document.getElementById(\'clusterContent\').innerHTML =\n    \'<div class="analyzing"><div class="loading-spinner"></div>Analyzing ticket clusters with Claude... ~30 seconds.</div>\';\n  fetch(\'/api/clusters\').then(r => r.json()).then(clusters => {\n    document.getElementById(\'clusterBtn\').disabled = false;\n    const gaps = clusters.filter(c => !c.has_kb_coverage);\n    document.getElementById(\'gapBadge\').textContent = gaps.length;\n    document.getElementById(\'statGaps\').textContent = gaps.length;\n    document.getElementById(\'clusterContent\').innerHTML =\n      \'<table class="data-table"><thead><tr><th>Topic Cluster</th><th>Est. Tickets</th><th>KB Coverage</th><th>Priority</th><th></th></tr></thead><tbody>\' +\n      clusters.map(c =>\n        \'<tr style="cursor:pointer;" onclick=\\\'showClusterDetail(\' + JSON.stringify(c).replace(/\'/g, "\\\\\'") + \')\\\'>\' +\n        \'<td><div style="font-weight:500;">\' + c.name + \'</div><div style="font-size:11px;color:var(--muted);margin-top:2px;">\' + c.description + \'</div></td>\' +\n        \'<td><span style="font-family:var(--mono);color:var(--amber);">~\' + c.ticket_count_estimate + \'</span></td>\' +\n        \'<td>\' + (c.has_kb_coverage ? \'<span class="badge badge-covered">✓ Covered</span>\' : \'<span class="badge badge-gap">✗ Gap</span>\') + \'</td>\' +\n        \'<td><span class="badge badge-\' + c.gap_severity + \'">\' + c.gap_severity + \'</span></td>\' +\n        \'<td><button class="btn" onclick="event.stopPropagation();draftArticle(\\\'\' + c.id + \'\\\')">Draft</button></td>\' +\n        \'</tr>\'\n      ).join(\'\') + \'</tbody></table>\';\n  }).catch(e => {\n    document.getElementById(\'clusterBtn\').disabled = false;\n    document.getElementById(\'clusterContent\').innerHTML = \'<div class="empty"><div>Error: \' + e.message + \'</div></div>\';\n  });\n}\n\nfunction showClusterDetail(c) {\n  const examples = (c.representative_subjects || []).map(s =>\n    \'<div class="list-item"><span class="list-bullet">›</span><span>\' + s + \'</span></div>\').join(\'\');\n  document.getElementById(\'detailTitle\').textContent = c.name;\n  document.getElementById(\'detailBody\').innerHTML =\n    \'<div><div style="font-size:12px;color:var(--muted);margin-bottom:8px;">DESCRIPTION</div><div>\' + c.description + \'</div></div>\' +\n    \'<div style="display:flex;gap:16px;">\' +\n      \'<div class="stat" style="flex:1;border:1px solid var(--border);border-radius:4px;"><div class="stat-value amber">~\' + c.ticket_count_estimate + \'</div><div class="stat-label">Est. Tickets</div></div>\' +\n      \'<div class="stat" style="flex:1;border:1px solid var(--border);border-radius:4px;"><div class="stat-value \' + (c.gap_severity===\'high\'?\'red\':c.gap_severity===\'medium\'?\'amber\':\'green\') + \'">\' + c.gap_severity.toUpperCase() + \'</div><div class="stat-label">Priority</div></div>\' +\n    \'</div>\' +\n    \'<div><div style="font-size:12px;color:var(--muted);margin-bottom:8px;">KB COVERAGE</div>\' +\n      (c.has_kb_coverage ? \'<span class="badge badge-covered">✓ \' + c.kb_article_title + \'</span>\' : \'<span class="badge badge-gap">✗ No existing article</span>\') +\n    \'</div>\' +\n    (examples ? \'<div><div style="font-size:12px;color:var(--muted);margin-bottom:8px;">EXAMPLE TICKETS</div>\' + examples + \'</div>\' : \'\') +\n    (!c.has_kb_coverage ? \'<div><button class="btn btn-primary" onclick="draftArticle(\\\'\' + c.id + \'\\\')">Generate Article Draft</button></div>\' : \'\');\n  openDetail();\n}\n\nfunction draftArticle(clusterId) {\n  document.getElementById(\'detailTitle\').textContent = \'Generating Draft...\';\n  document.getElementById(\'detailBody\').innerHTML = \'<div class="analyzing"><div class="loading-spinner"></div>Claude is writing the article...</div>\';\n  openDetail();\n  fetch(\'/api/clusters/\' + clusterId + \'/draft\').then(r => r.json()).then(d => {\n    const sections = (d.sections||[]).map(s =>\n      \'<div class="draft-section"><h3>\' + s.heading + \'</h3><p>\' + s.content + \'</p></div>\').join(\'\');\n    const trouble = (d.troubleshooting||[]).map(t =>\n      \'<div class="list-item"><span class="list-bullet" style="color:var(--red);">?</span><div><strong>\' + t.problem + \'</strong><br><span style="color:var(--muted);">\' + t.solution + \'</span></div></div>\').join(\'\');\n    document.getElementById(\'detailTitle\').textContent = d.title || \'Draft Article\';\n    document.getElementById(\'detailBody\').innerHTML =\n      \'<div class="draft-section"><h3>Introduction</h3><p>\' + (d.intro||\'\') + \'</p></div>\' +\n      sections +\n      (trouble ? \'<div class="draft-section"><h3>Troubleshooting</h3>\' + trouble + \'</div>\' : \'\') +\n      \'<div><button class="btn" onclick="copyDraft()">Copy as Markdown</button></div>\';\n    window._lastDraft = d;\n  });\n}\n\nfunction copyDraft() {\n  const d = window._lastDraft;\n  if (!d) return;\n  let md = \'# \' + d.title + \'\\n\\n\' + d.intro + \'\\n\\n\';\n  (d.sections||[]).forEach(s => { md += \'## \' + s.heading + \'\\n\\n\' + s.content + \'\\n\\n\'; });\n  if ((d.troubleshooting||[]).length) {\n    md += \'## Troubleshooting\\n\\n\';\n    d.troubleshooting.forEach(t => { md += \'**\' + t.problem + \'**\\n\' + t.solution + \'\\n\\n\'; });\n  }\n  navigator.clipboard.writeText(md).then(() => alert(\'Copied to clipboard!\'));\n}\n\nfunction loadArticles() {\n  if (articlesLoaded) return;\n  fetch(\'/api/articles\').then(r => r.json()).then(articles => {\n    articlesLoaded = true;\n    document.getElementById(\'articlesContent\').innerHTML =\n      \'<table class="data-table"><thead><tr><th>Article</th><th>Last Updated</th><th></th></tr></thead><tbody>\' +\n      articles.map(a =>\n        \'<tr style="cursor:pointer;" onclick="analyzeArticle(\' + a.id + \', \\\'\' + a.title.replace(/\'/g,"\\\\\'") + \'\\\')">\' +\n        \'<td><div style="font-weight:500;">\' + a.title + \'</div>\' + (a.draft ? \'<span style="font-size:10px;color:var(--amber);">DRAFT</span>\' : \'\') + \'</td>\' +\n        \'<td style="font-family:var(--mono);font-size:11px;color:var(--muted);">\' + (a.updated_at||\'\').split(\'T\')[0] + \'</td>\' +\n        \'<td><button class="btn" onclick="event.stopPropagation();analyzeArticle(\' + a.id + \',\\\'\' + a.title.replace(/\'/g,"\\\\\'") + \'\\\')">Analyze</button></td>\' +\n        \'</tr>\'\n      ).join(\'\') + \'</tbody></table>\';\n  });\n}\n\nfunction analyzeArticle(id, title) {\n  document.getElementById(\'detailTitle\').textContent = title;\n  document.getElementById(\'detailBody\').innerHTML = \'<div class="analyzing"><div class="loading-spinner"></div>Analyzing against ticket data...</div>\';\n  openDetail();\n  fetch(\'/api/articles/\' + id + \'/analyze\').then(r => r.json()).then(d => {\n    const scoreClass = d.quality_score >= 70 ? \'score-high\' : d.quality_score >= 40 ? \'score-mid\' : \'score-low\';\n    const issues = (d.issues||[]).map(i => \'<div class="list-item"><span class="list-bullet" style="color:var(--red);">✗</span><span>\' + i + \'</span></div>\').join(\'\');\n    const suggestions = (d.suggestions||[]).map(s => \'<div class="list-item"><span class="list-bullet" style="color:var(--green);">→</span><span>\' + s + \'</span></div>\').join(\'\');\n    const questions = (d.unanswered_questions||[]).map(q => \'<div class="list-item"><span class="list-bullet" style="color:var(--amber);">?</span><span>\' + q + \'</span></div>\').join(\'\');\n    document.getElementById(\'detailBody\').innerHTML =\n      \'<div style="display:flex;align-items:center;gap:16px;"><div class="score-ring \' + scoreClass + \'">\' + d.quality_score + \'</div>\' +\n      \'<div><div style="font-weight:500;">Quality Score</div><div style="font-size:12px;color:var(--muted);">\' + d.related_ticket_count + \' related tickets found</div></div></div>\' +\n      \'<div style="font-size:13px;color:var(--muted);padding:12px;background:var(--surface2);border-radius:4px;">\' + d.summary + \'</div>\' +\n      (issues ? \'<div><div style="font-size:12px;color:var(--muted);margin-bottom:8px;">ISSUES FOUND</div>\' + issues + \'</div>\' : \'\') +\n      (suggestions ? \'<div><div style="font-size:12px;color:var(--muted);margin-bottom:8px;">SUGGESTIONS</div>\' + suggestions + \'</div>\' : \'\') +\n      (questions ? \'<div><div style="font-size:12px;color:var(--muted);margin-bottom:8px;">UNANSWERED CUSTOMER QUESTIONS</div>\' + questions + \'</div>\' : \'\');\n  });\n}\n\nfunction runOutdatedAnalysis() {\n  if (!appReady) return;\n  document.getElementById(\'outdatedBtn\').disabled = true;\n  document.getElementById(\'outdatedContent\').innerHTML =\n    \'<div class="analyzing"><div class="loading-spinner"></div>Scanning tickets for documentation complaints...</div>\';\n  fetch(\'/api/outdated\').then(r => r.json()).then(items => {\n    document.getElementById(\'outdatedBtn\').disabled = false;\n    document.getElementById(\'statOutdated\').textContent = items.length;\n    if (!items.length) {\n      document.getElementById(\'outdatedContent\').innerHTML =\n        \'<div class="empty"><div class="empty-icon">✓</div><div>No strong signals of outdated articles found.</div></div>\';\n      return;\n    }\n    document.getElementById(\'outdatedContent\').innerHTML =\n      \'<table class="data-table"><thead><tr><th>Article</th><th>Confidence</th><th>Evidence Tickets</th></tr></thead><tbody>\' +\n      items.map(item =>\n        \'<tr><td><div style="font-weight:500;">\' + item.article_title + \'</div>\' +\n        \'<div style="font-size:11px;color:var(--muted);margin-top:4px;">\' + item.reason + \'</div></td>\' +\n        \'<td><span class="badge badge-\' + item.confidence + \'">\' + item.confidence + \'</span></td>\' +\n        \'<td style="font-family:var(--mono);font-size:11px;color:var(--muted);">\' + (item.evidence_ticket_ids||[]).join(\', \') + \'</td></tr>\'\n      ).join(\'\') + \'</tbody></table>\';\n  }).catch(() => { document.getElementById(\'outdatedBtn\').disabled = false; });\n}\n\nfunction openDetail() { document.getElementById(\'detailPanel\').classList.add(\'open\'); }\nfunction closeDetail() { document.getElementById(\'detailPanel\').classList.remove(\'open\'); }\n\npoll();\n</script>\n</body>\n</html>\n'


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": state["load_status"],
        "message": state["load_message"],
        "ticket_count": len(state["tickets"]),
        "article_count": len(state["kb_articles"])
    })


@app.route("/api/reload", methods=["POST"])
def api_reload():
    if state["loading"]:
        return jsonify({"error": "Already loading"}), 400
    state["analysis_cache"] = {}
    thread = threading.Thread(target=load_tickets, daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/overview")
def api_overview():
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503

    tickets = state["tickets"]
    status_counts = {}
    for t in tickets:
        s = t.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    return jsonify({
        "ticket_count": len(tickets),
        "article_count": len(state["kb_articles"]),
        "status_breakdown": status_counts,
        "tag_frequency": analyze_high_volume(tickets)[:15]
    })


@app.route("/api/clusters")
def api_clusters():
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503
    clusters = analyze_topic_clusters(state["tickets"], state["kb_articles"])
    return jsonify(clusters)


@app.route("/api/articles")
def api_articles():
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503
    articles = state["kb_articles"]
    # Return lightweight list
    return jsonify([{
        "id": a.get("id"),
        "title": a.get("title"),
        "html_url": a.get("html_url"),
        "updated_at": a.get("updated_at"),
        "draft": a.get("draft"),
    } for a in articles])


@app.route("/api/articles/<int:article_id>/analyze")
def api_analyze_article(article_id):
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503
    article = next((a for a in state["kb_articles"] if a.get("id") == article_id), None)
    if not article:
        return jsonify({"error": "Article not found"}), 404
    result = analyze_article_improvement(article)
    return jsonify(result)


@app.route("/api/outdated")
def api_outdated():
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503
    result = analyze_outdated_articles(state["kb_articles"], state["tickets"])
    return jsonify(result)


@app.route("/api/clusters/<cluster_id>/draft")
def api_draft_article(cluster_id):
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503
    clusters = state["analysis_cache"].get("clusters", [])
    cluster = next((c for c in clusters if c.get("id") == cluster_id), None)
    if not cluster:
        return jsonify({"error": "Cluster not found -- run gap analysis first"}), 404
    result = generate_new_article_draft(cluster)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Startup -- use @app.before_request so it fires in whichever worker
# receives the first request, regardless of gunicorn worker count
# ---------------------------------------------------------------------------

_startup_done = False
_startup_lock = threading.Lock()

@app.before_request
def startup_once():
    global _startup_done
    if not _startup_done:
        with _startup_lock:
            if not _startup_done:
                _startup_done = True
                thread = threading.Thread(target=load_tickets, daemon=True)
                thread.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)