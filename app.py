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
    "org_names": {},    # org_id -> name
    "user_names": {},   # user_id -> name
    "loading": False,
    "load_status": "idle",  # idle | loading | ready | error
    "load_message": "",
    "analysis_cache": {},
    "filters": {
        "date_from": None,
        "date_to": None,
        "tags_include": [],
        "tags_exclude": [],
        "assignee_ids": [],
        "org_ids": []
    }
}

# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def get_filtered_tickets():
    """Return tickets filtered by current global filter state."""
    tickets = state["tickets"]
    f = state["filters"]

    result = []
    for t in tickets:
        created = t.get("created_at", "")

        if f["date_from"] and created < f["date_from"]:
            continue
        if f["date_to"] and created > f["date_to"]:
            continue

        ticket_tags = set(t.get("tags") or [])
        if f["tags_include"] and not ticket_tags.intersection(f["tags_include"]):
            continue
        if f["tags_exclude"] and ticket_tags.intersection(f["tags_exclude"]):
            continue

        if f["assignee_ids"] and t.get("assignee_id") not in f["assignee_ids"]:
            continue
        if f["org_ids"] and t.get("organization_id") not in f["org_ids"]:
            continue

        result.append(t)
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_zendesk_lookup(resource_key, path):
    """Fetch all pages of a Zendesk list endpoint and return items list."""
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
    items = []
    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/{path}"
    while url:
        try:
            r = requests.get(url, auth=auth, timeout=30)
            if r.status_code != 200:
                app.logger.warning(f"Zendesk {path} returned {r.status_code}: {r.text[:200]}")
                break
            data = r.json()
            items.extend(data.get(resource_key, []))
            url = data.get("next_page")
        except Exception as e:
            app.logger.warning(f"Zendesk lookup error for {path}: {e}")
            break
    return items


def load_tickets():
    """Download and parse the ticket export from GitHub Releases."""
    state["loading"] = True
    state["load_status"] = "loading"
    state["load_message"] = "Downloading ticket data..."

    try:
        state["load_message"] = f"Downloading ticket export (~330MB, this takes a minute)..."
        resp = requests.get(
            DATA_URL,
            timeout=300,
            stream=True,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; vs-kb-analyzer/1.0)"}
        )
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code} fetching data file (final URL: {resp.url})")
        state["load_message"] = "Parsing ticket data..."
        data = json.loads(resp.content)
        all_tickets = data.get("tickets", [])

        # Filter out automation and smoke test tickets
        def is_noise(ticket):
            subject = (ticket.get("subject") or "").lower()
            requester = (ticket.get("requester", {}) or {})
            requester_email = (requester.get("email") or "").lower()
            requester_name = (requester.get("name") or "").lower()
            tags = [t.lower() for t in (ticket.get("tags") or [])]
            comments = ticket.get("comments") or []
            comment_bodies = " ".join((c.get("body") or "").lower() for c in comments)

            if "smoke test" in subject or "smoketest" in subject:
                return True
            if "smoke test" in comment_bodies or "smoketest" in comment_bodies:
                return True
            if "smoke_test" in tags or "smoketest" in tags:
                return True
            if "support automation" in requester_name:
                return True
            if "support+automation" in requester_email or "automation@vergesense" in requester_email:
                return True
            return False

        state["tickets"] = [t for t in all_tickets if not is_noise(t)]
        filtered = len(all_tickets) - len(state["tickets"])
        state["load_message"] = f"Loaded {len(state['tickets'])} tickets ({filtered} automation/test tickets filtered)."

        # Fetch org and user names from Zendesk for human-readable filter dropdowns
        state["load_message"] = "Fetching organization names..."
        orgs = fetch_zendesk_lookup("organizations", "organizations.json?per_page=100")
        state["org_names"] = {o["id"]: o["name"] for o in orgs}

        state["load_message"] = "Fetching agent names..."
        users = fetch_zendesk_lookup("users", "users.json?role=agent&per_page=100")
        state["user_names"] = {u["id"]: u["name"] for u in users}
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
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return msg.content[0].text


def call_claude_json(system, user, max_tokens=2000):
    """Claude call that returns parsed JSON."""
    system += "\n\nRespond ONLY with valid JSON. No markdown, no backticks, no explanation."
    try:
        text = call_claude(system, user, max_tokens)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        app.logger.error(f"call_claude_json failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def analyze_topic_clusters(tickets, kb_articles):
    """
    Cluster tickets into topics, count frequency, and identify KB gaps.
    Returns list of cluster dicts.
    """
    # Cache key includes filter fingerprint so filtered runs don't return unfiltered results
    f = state["filters"]
    cache_key = f"clusters:{f['date_from']}:{f['date_to']}:{sorted(f['tags_include'])}:{sorted(f['tags_exclude'])}:{sorted(f['assignee_ids'])}:{sorted(f['org_ids'])}"
    if cache_key in state["analysis_cache"]:
        return state["analysis_cache"][cache_key]

    if not tickets:
        return []

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
    for t in get_filtered_tickets():
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
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html")) as f:
        return f.read()



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


@app.route("/api/filter-options")
def api_filter_options():
    """Return unique assignees, orgs, and tags for filter dropdowns."""
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503

    assignees = {}
    orgs = {}
    tags = set()

    for t in state["tickets"]:
        aid = t.get("assignee_id")
        if aid:
            aname = state["user_names"].get(aid) or str(aid)
            assignees[aid] = aname

        oid = t.get("organization_id")
        if oid:
            oname = state["org_names"].get(oid) or str(oid)
            orgs[oid] = oname

        for tag in (t.get("tags") or []):
            tags.add(tag)

    return jsonify({
        "assignees": [{"id": k, "name": v} for k, v in sorted(assignees.items(), key=lambda x: x[1])],
        "orgs": [{"id": k, "name": v} for k, v in sorted(orgs.items(), key=lambda x: x[1])],
        "tags": sorted(list(tags))
    })


@app.route("/api/filters", methods=["GET"])
def api_get_filters():
    return jsonify(state["filters"])


@app.route("/api/filters", methods=["POST"])
def api_set_filters():
    data = request.json or {}
    state["filters"]["date_from"] = data.get("date_from") or None
    state["filters"]["date_to"] = data.get("date_to") or None
    state["filters"]["tags_include"] = data.get("tags_include") or []
    state["filters"]["tags_exclude"] = data.get("tags_exclude") or []
    state["filters"]["assignee_ids"] = [int(x) for x in data.get("assignee_ids") or []]
    state["filters"]["org_ids"] = [int(x) for x in data.get("org_ids") or []]
    # Clear analysis cache so results re-run against new filter set
    state["analysis_cache"] = {}
    filtered = get_filtered_tickets()
    return jsonify({"ok": True, "matched_tickets": len(filtered)})


@app.route("/api/overview")
def api_overview():
    if state["load_status"] != "ready":
        return jsonify({"error": "Data not ready"}), 503

    tickets = get_filtered_tickets()
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
    clusters = analyze_topic_clusters(get_filtered_tickets(), state["kb_articles"])
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
    result = analyze_outdated_articles(state["kb_articles"], get_filtered_tickets())
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