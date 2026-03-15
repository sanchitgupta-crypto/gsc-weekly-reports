"""
SquadStack.ai - Weekly Google Search Console Report Generator
=============================================================
Pulls GSC data, computes WoW/MoM deltas, segments by Voice AI keyword
clusters, identifies wins and issues, and emails a formatted HTML digest.

SETUP:
1. Enable GSC API in Google Cloud Console
2. Create a service account and download credentials JSON
3. Add the service account email as a user in GSC for squadstack.ai
4. pip install google-api-python-client google-auth google-auth-oauthlib pandas jinja2
5. Set environment variables (see CONFIG section below)

USAGE:
  python gsc_weekly_report.py                    # Run for last 7 days
  python gsc_weekly_report.py --weeks 2          # Compare last 2 weeks
  python gsc_weekly_report.py --monthly           # Monthly report
  python gsc_weekly_report.py --dry-run           # Print to console, no email
"""

import os
import json
import argparse
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import defaultdict

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from jinja2 import Template

# ============================================================
# CONFIG - Set these as environment variables or edit directly
# ============================================================
SITE_URL = os.environ.get("GSC_SITE_URL", "sc-domain:squadstack.ai")
CREDENTIALS_FILE = os.environ.get("GSC_CREDENTIALS", "gsc-credentials.json")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")  # sender email
SMTP_PASS = os.environ.get("SMTP_PASS", "")  # app password
REPORT_RECIPIENTS = os.environ.get(
    "REPORT_RECIPIENTS",
    "sanchit@squadstack.com"
).split(",")

# ============================================================
# KEYWORD CLUSTERS - Voice AI specific for SquadStack
# ============================================================
KEYWORD_CLUSTERS = {
    "Brand": [
        "squadstack", "squad stack", "squadstack.ai"
    ],
    "Voice AI (Core)": [
        "voice ai", "ai voice", "voice agent", "ai calling",
        "voice bot", "voicebot", "ai voice agent",
        "humanoid ai", "humanoid agent", "voice automation",
        "ai phone calls", "automated calling", "ai cold calling",
        "conversational ai voice"
    ],
    "Contact Center AI": [
        "contact center ai", "call center ai", "ai call center",
        "ai contact center", "ai telecalling", "telecalling ai",
        "ai bpo", "contact center automation", "call center automation",
        "ai ivr", "intelligent ivr"
    ],
    "Sales AI / Revenue AI": [
        "ai sales", "sales ai", "ai for sales", "revenue ai",
        "ai sales agent", "ai sales advisor", "ai sdr",
        "ai inside sales", "sales automation ai",
        "lead conversion ai", "ai lead qualification"
    ],
    "Industry (BFSI)": [
        "ai insurance sales", "ai lending", "ai loan calling",
        "ai for banking", "ai for insurance", "bfsi ai",
        "ai collections", "loan recovery ai",
        "ai for nbfc", "fintech ai calling"
    ],
    "Industry (Consumer / E-commerce)": [
        "ai for ecommerce", "ecommerce ai calling",
        "ai customer engagement", "ai reactivation",
        "ai for edtech", "ai for real estate",
        "ai for healthcare sales"
    ],
    "WhatsApp / Omnichannel": [
        "whatsapp ai", "whatsapp bot", "whatsapp automation",
        "omnichannel ai", "ai whatsapp agent",
        "whatsapp business ai"
    ],
    "Speech Tech (Arth/Goonj)": [
        "speech to text india", "indian stt", "hindi stt",
        "text to speech india", "indian tts", "hindi tts",
        "indian voice cloning", "vernacular ai",
        "indian language ai", "multilingual voice ai"
    ],
    "Competitor Adjacent": [
        "exotel ai", "ozonetel ai", "knowlarity ai",
        "ameyo ai", "leadsquared calling", "freshcaller ai",
        "yellow.ai voice", "haptik voice", "verloop voice",
        "sarvam ai", "deepgram india"
    ],
    "Use Case Specific": [
        "ai quality monitoring", "call quality ai",
        "ai a/b testing calls", "ai lead scoring",
        "ai dialer", "predictive dialer ai",
        "ai outbound calling", "ai inbound calling"
    ]
}


def get_gsc_service():
    """Authenticate and return GSC service object."""
    scopes = ["https://www.googleapis.com/auth/webmasters.readonly"]
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=scopes
    )
    return build("searchconsole", "v1", credentials=credentials)


def fetch_gsc_data(service, start_date, end_date, dimensions=None):
    """Fetch data from GSC API with pagination."""
    if dimensions is None:
        dimensions = ["query", "page", "date"]

    all_rows = []
    start_row = 0
    row_limit = 25000

    while True:
        request_body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions,
            "rowLimit": row_limit,
            "startRow": start_row,
            "dataState": "final"
        }

        response = service.searchanalytics().query(
            siteUrl=SITE_URL, body=request_body
        ).execute()

        rows = response.get("rows", [])
        if not rows:
            break

        for row in rows:
            keys = row["keys"]
            record = {
                "clicks": row["clicks"],
                "impressions": row["impressions"],
                "ctr": row["ctr"],
                "position": row["position"]
            }
            for i, dim in enumerate(dimensions):
                record[dim] = keys[i]
            all_rows.append(record)

        start_row += row_limit
        if len(rows) < row_limit:
            break

    return pd.DataFrame(all_rows)


def classify_query(query, clusters):
    """Assign a query to a keyword cluster. Returns cluster name or 'Other'."""
    q = query.lower().strip()
    for cluster_name, keywords in clusters.items():
        for kw in keywords:
            if kw in q:
                return cluster_name
    return "Other"


def compute_period_metrics(df, group_col="query"):
    """Aggregate metrics for a period grouped by a column."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby(group_col).agg(
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        ctr=("ctr", "mean"),
        avg_position=("position", "mean")
    ).reset_index()


def compute_deltas(current, previous, join_col="query", label="WoW"):
    """Compute period-over-period deltas."""
    if current.empty or previous.empty:
        return current

    merged = current.merge(
        previous, on=join_col, suffixes=("", "_prev"), how="left"
    )

    for metric in ["clicks", "impressions"]:
        prev_col = f"{metric}_prev"
        delta_col = f"{metric}_{label}_delta"
        pct_col = f"{metric}_{label}_pct"

        merged[delta_col] = merged[metric] - merged.get(prev_col, 0).fillna(0)
        merged[pct_col] = merged.apply(
            lambda r: (
                (r[metric] - r.get(prev_col, 0)) / r.get(prev_col, 1) * 100
                if pd.notna(r.get(prev_col)) and r.get(prev_col, 0) > 0
                else None
            ), axis=1
        )

    merged["position_change"] = (
        merged.get("avg_position_prev", pd.Series()).fillna(0)
        - merged["avg_position"]
    )

    return merged


def get_top_movers(df, metric="clicks", n=15, direction="up"):
    """Get top N queries by metric change."""
    delta_col = f"{metric}_WoW_delta"
    if delta_col not in df.columns:
        return pd.DataFrame()

    if direction == "up":
        return df.nlargest(n, delta_col)
    else:
        return df.nsmallest(n, delta_col)


def compute_cluster_summary(df, clusters):
    """Summarize metrics by keyword cluster."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["cluster"] = df["query"].apply(lambda q: classify_query(q, clusters))

    summary = df.groupby("cluster").agg(
        queries=("query", "nunique"),
        total_clicks=("clicks", "sum"),
        total_impressions=("impressions", "sum"),
        avg_position=("avg_position", "mean") if "avg_position" in df.columns else ("position", "mean"),
        avg_ctr=("ctr", "mean")
    ).reset_index().sort_values("total_clicks", ascending=False)

    return summary


def compute_page_performance(df):
    """Group by page URL and compute metrics."""
    if df.empty or "page" not in df.columns:
        return pd.DataFrame()

    pages = df.groupby("page").agg(
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        avg_ctr=("ctr", "mean"),
        avg_position=("position", "mean"),
        unique_queries=("query", "nunique")
    ).reset_index().sort_values("clicks", ascending=False)

    # Categorize pages
    def categorize_page(url):
        url = url.lower()
        if "/blog/" in url or "/resources/" in url:
            return "Blog / Resources"
        elif any(x in url for x in ["/product", "/platform", "/voice-ai", "/humanoid"]):
            return "Product Pages"
        elif any(x in url for x in ["/case-study", "/customer", "/success"]):
            return "Case Studies"
        elif any(x in url for x in ["/industry", "/bfsi", "/insurance", "/lending"]):
            return "Industry Pages"
        elif any(x in url for x in ["/demo", "/contact", "/pricing", "/get-started"]):
            return "Conversion Pages"
        elif url.rstrip("/").endswith("squadstack.ai") or url.endswith("/"):
            return "Homepage"
        else:
            return "Other"

    pages["page_type"] = pages["page"].apply(categorize_page)
    return pages


def identify_new_queries(current_queries, previous_queries):
    """Find queries that appeared this period but not last."""
    new = set(current_queries) - set(previous_queries)
    return list(new)


def identify_lost_queries(current_queries, previous_queries, min_prev_clicks=3):
    """Find queries that had clicks last period but disappeared."""
    lost = set(previous_queries) - set(current_queries)
    return list(lost)


def generate_insights(data):
    """Auto-generate key insights from the data."""
    insights = {"wins": [], "needs_attention": [], "opportunities": []}

    # Cluster performance
    if "cluster_summary" in data and not data["cluster_summary"].empty:
        cs = data["cluster_summary"]
        top_cluster = cs.iloc[0]
        insights["wins"].append(
            f"'{top_cluster['cluster']}' cluster led with {int(top_cluster['total_clicks'])} clicks "
            f"from {int(top_cluster['total_impressions'])} impressions"
        )

        # Low CTR clusters with high impressions (opportunity)
        high_imp_low_ctr = cs[
            (cs["total_impressions"] > cs["total_impressions"].median())
            & (cs["avg_ctr"] < cs["avg_ctr"].median())
        ]
        for _, row in high_imp_low_ctr.iterrows():
            insights["opportunities"].append(
                f"'{row['cluster']}' has {int(row['total_impressions'])} impressions "
                f"but only {row['avg_ctr']:.1%} CTR. Title/meta optimization could unlock clicks."
            )

    # Top movers
    if "top_gainers" in data and not data["top_gainers"].empty:
        top = data["top_gainers"].iloc[0]
        insights["wins"].append(
            f"Top gaining query: '{top['query']}' added {int(top.get('clicks_WoW_delta', 0))} clicks WoW"
        )

    if "top_losers" in data and not data["top_losers"].empty:
        worst = data["top_losers"].iloc[0]
        delta = int(abs(worst.get("clicks_WoW_delta", 0)))
        insights["needs_attention"].append(
            f"Biggest drop: '{worst['query']}' lost {delta} clicks WoW"
        )

    # Position movements
    if "current_with_deltas" in data and not data["current_with_deltas"].empty:
        df = data["current_with_deltas"]
        if "position_change" in df.columns:
            improved = df[df["position_change"] > 2].shape[0]
            declined = df[df["position_change"] < -2].shape[0]
            if improved > 0:
                insights["wins"].append(
                    f"{improved} queries improved position by 2+ spots"
                )
            if declined > 0:
                insights["needs_attention"].append(
                    f"{declined} queries dropped position by 2+ spots"
                )

    # New queries
    if "new_queries" in data and data["new_queries"]:
        count = len(data["new_queries"])
        insights["wins"].append(
            f"{count} new queries entered the results this week"
        )

    # Lost queries
    if "lost_queries" in data and data["lost_queries"]:
        count = len(data["lost_queries"])
        insights["needs_attention"].append(
            f"{count} queries from last week no longer appearing"
        )

    # Page type performance
    if "page_performance" in data and not data["page_performance"].empty:
        pp = data["page_performance"]
        if "page_type" in pp.columns:
            by_type = pp.groupby("page_type").agg(
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum")
            ).sort_values("clicks", ascending=False)

            if not by_type.empty:
                top_type = by_type.index[0]
                insights["wins"].append(
                    f"'{top_type}' pages drove the most clicks ({int(by_type.iloc[0]['clicks'])})"
                )

    return insights


# ============================================================
# HTML EMAIL TEMPLATE
# ============================================================
EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
    h1 { color: #0f172a; font-size: 22px; border-bottom: 3px solid #16a34a; padding-bottom: 8px; }
    h2 { color: #334155; font-size: 17px; margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
    h3 { color: #475569; font-size: 15px; margin-top: 20px; }
    .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: 700; color: #0f172a; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    .up { color: #16a34a; }
    .down { color: #dc2626; }
    .neutral { color: #64748b; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
    th { background: #f1f5f9; color: #334155; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e2e8f0; }
    td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
    tr:hover td { background: #f8fafc; }
    .insight-box { border-radius: 8px; padding: 14px 18px; margin: 8px 0; }
    .win { background: #f0fdf4; border-left: 4px solid #16a34a; }
    .issue { background: #fef2f2; border-left: 4px solid #dc2626; }
    .opp { background: #fffbeb; border-left: 4px solid #f59e0b; }
    .insight-title { font-weight: 600; margin-bottom: 6px; }
    .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #94a3b8; }
    .cluster-bar { height: 8px; background: #16a34a; border-radius: 4px; display: inline-block; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-brand { background: #dbeafe; color: #1e40af; }
    .tag-nonbrand { background: #f0fdf4; color: #166534; }
</style>
</head>
<body>

<h1>SquadStack.ai GSC Weekly Report</h1>
<p style="color: #64748b; font-size: 14px;">
    {{ period_label }} | Generated {{ generated_at }}
</p>

<!-- TOP-LINE METRICS -->
<div class="summary-grid">
    <div class="metric-card">
        <div class="metric-value">{{ "{:,}".format(total_clicks) }}</div>
        <div class="metric-label">Clicks</div>
        <div class="metric-delta {{ 'up' if clicks_delta > 0 else 'down' if clicks_delta < 0 else 'neutral' }}">
            {{ "{:+,.0f}".format(clicks_delta) }} ({{ "{:+.1f}%".format(clicks_pct) if clicks_pct else "N/A" }}) WoW
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{{ "{:,}".format(total_impressions) }}</div>
        <div class="metric-label">Impressions</div>
        <div class="metric-delta {{ 'up' if imp_delta > 0 else 'down' if imp_delta < 0 else 'neutral' }}">
            {{ "{:+,.0f}".format(imp_delta) }} ({{ "{:+.1f}%".format(imp_pct) if imp_pct else "N/A" }}) WoW
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{{ "{:.1%}".format(avg_ctr) }}</div>
        <div class="metric-label">Avg CTR</div>
        <div class="metric-delta {{ 'up' if ctr_delta > 0 else 'down' if ctr_delta < 0 else 'neutral' }}">
            {{ "{:+.2f}pp".format(ctr_delta * 100) }} WoW
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{{ "{:.1f}".format(avg_position) }}</div>
        <div class="metric-label">Avg Position</div>
        <div class="metric-delta {{ 'up' if pos_delta > 0 else 'down' if pos_delta < 0 else 'neutral' }}">
            {{ "{:+.1f}".format(pos_delta) }} WoW
        </div>
    </div>
</div>

<!-- KEY INSIGHTS -->
<h2>Key Insights</h2>

{% if insights.wins %}
<div class="insight-box win">
    <div class="insight-title">What Went Well</div>
    <ul style="margin: 4px 0; padding-left: 18px;">
    {% for w in insights.wins %}
        <li>{{ w }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}

{% if insights.needs_attention %}
<div class="insight-box issue">
    <div class="insight-title">Needs Attention</div>
    <ul style="margin: 4px 0; padding-left: 18px;">
    {% for n in insights.needs_attention %}
        <li>{{ n }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}

{% if insights.opportunities %}
<div class="insight-box opp">
    <div class="insight-title">Opportunities</div>
    <ul style="margin: 4px 0; padding-left: 18px;">
    {% for o in insights.opportunities %}
        <li>{{ o }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}

<!-- BRANDED vs NON-BRANDED -->
<h2>Branded vs Non-Branded</h2>
<table>
    <tr>
        <th>Segment</th>
        <th>Clicks</th>
        <th>Impressions</th>
        <th>CTR</th>
        <th>Avg Position</th>
    </tr>
    <tr>
        <td><span class="tag tag-brand">Branded</span></td>
        <td>{{ "{:,}".format(branded_clicks) }}</td>
        <td>{{ "{:,}".format(branded_impressions) }}</td>
        <td>{{ "{:.1%}".format(branded_ctr) }}</td>
        <td>{{ "{:.1f}".format(branded_position) }}</td>
    </tr>
    <tr>
        <td><span class="tag tag-nonbrand">Non-Branded</span></td>
        <td>{{ "{:,}".format(nonbranded_clicks) }}</td>
        <td>{{ "{:,}".format(nonbranded_impressions) }}</td>
        <td>{{ "{:.1%}".format(nonbranded_ctr) }}</td>
        <td>{{ "{:.1f}".format(nonbranded_position) }}</td>
    </tr>
</table>

<!-- KEYWORD CLUSTER PERFORMANCE -->
<h2>Performance by Keyword Cluster</h2>
<table>
    <tr>
        <th>Cluster</th>
        <th>Queries</th>
        <th>Clicks</th>
        <th>Impressions</th>
        <th>CTR</th>
        <th>Avg Pos</th>
    </tr>
    {% for _, row in cluster_summary.iterrows() %}
    <tr>
        <td><strong>{{ row.cluster }}</strong></td>
        <td>{{ row.queries }}</td>
        <td>{{ "{:,}".format(row.total_clicks | int) }}</td>
        <td>{{ "{:,}".format(row.total_impressions | int) }}</td>
        <td>{{ "{:.1%}".format(row.avg_ctr) }}</td>
        <td>{{ "{:.1f}".format(row.avg_position) }}</td>
    </tr>
    {% endfor %}
</table>

<!-- TOP GAINING QUERIES -->
<h2>Top Gaining Queries (WoW)</h2>
<table>
    <tr>
        <th>Query</th>
        <th>Clicks</th>
        <th>Change</th>
        <th>Impressions</th>
        <th>Position</th>
    </tr>
    {% for _, row in top_gainers.head(10).iterrows() %}
    <tr>
        <td>{{ row.query }}</td>
        <td>{{ "{:,}".format(row.clicks | int) }}</td>
        <td class="up">{{ "{:+,.0f}".format(row.clicks_WoW_delta) }}</td>
        <td>{{ "{:,}".format(row.impressions | int) }}</td>
        <td>{{ "{:.1f}".format(row.avg_position) }}</td>
    </tr>
    {% endfor %}
</table>

<!-- TOP DECLINING QUERIES -->
<h2>Top Declining Queries (WoW)</h2>
<table>
    <tr>
        <th>Query</th>
        <th>Clicks</th>
        <th>Change</th>
        <th>Impressions</th>
        <th>Position</th>
    </tr>
    {% for _, row in top_losers.head(10).iterrows() %}
    <tr>
        <td>{{ row.query }}</td>
        <td>{{ "{:,}".format(row.clicks | int) }}</td>
        <td class="down">{{ "{:+,.0f}".format(row.clicks_WoW_delta) }}</td>
        <td>{{ "{:,}".format(row.impressions | int) }}</td>
        <td>{{ "{:.1f}".format(row.avg_position) }}</td>
    </tr>
    {% endfor %}
</table>

<!-- PAGE PERFORMANCE BY TYPE -->
<h2>Page Performance by Type</h2>
<table>
    <tr>
        <th>Page Type</th>
        <th>Pages</th>
        <th>Clicks</th>
        <th>Impressions</th>
        <th>Avg CTR</th>
    </tr>
    {% for _, row in page_type_summary.iterrows() %}
    <tr>
        <td><strong>{{ row.page_type }}</strong></td>
        <td>{{ row.pages }}</td>
        <td>{{ "{:,}".format(row.clicks | int) }}</td>
        <td>{{ "{:,}".format(row.impressions | int) }}</td>
        <td>{{ "{:.1%}".format(row.avg_ctr) }}</td>
    </tr>
    {% endfor %}
</table>

<!-- TOP PAGES -->
<h2>Top 15 Pages by Clicks</h2>
<table>
    <tr>
        <th>Page</th>
        <th>Clicks</th>
        <th>Impressions</th>
        <th>CTR</th>
        <th>Queries</th>
    </tr>
    {% for _, row in top_pages.head(15).iterrows() %}
    <tr>
        <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            <a href="{{ row.page }}" style="color:#2563eb; text-decoration:none;">{{ row.page | replace("https://www.squadstack.ai", "") | replace("https://squadstack.ai", "") }}</a>
        </td>
        <td>{{ "{:,}".format(row.clicks | int) }}</td>
        <td>{{ "{:,}".format(row.impressions | int) }}</td>
        <td>{{ "{:.1%}".format(row.avg_ctr) }}</td>
        <td>{{ row.unique_queries }}</td>
    </tr>
    {% endfor %}
</table>

<!-- NEW QUERIES -->
{% if has_new_queries %}
<h2>New Queries This Week ({{ new_queries | length }})</h2>
<p style="font-size: 13px; color: #475569;">
    {% for q in new_queries[:30] %}
    <span style="display:inline-block; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:4px; padding:2px 8px; margin:2px; font-size:12px;">{{ q }}</span>
    {% endfor %}
    {% if new_queries | length > 30 %}
    <span style="color:#94a3b8;">... and {{ new_queries | length - 30 }} more</span>
    {% endif %}
</p>
{% endif %}

<!-- POSITION 4-10 OPPORTUNITIES -->
{% if striking_distance is not none %}
<h2>Striking Distance (Position 4-10, High Impressions)</h2>
<p style="font-size: 13px; color: #64748b; margin-bottom: 8px;">
    These queries rank on page 1 but below position 3. Small position improvements here unlock disproportionate click gains.
</p>
<table>
    <tr>
        <th>Query</th>
        <th>Position</th>
        <th>Impressions</th>
        <th>Clicks</th>
        <th>CTR</th>
    </tr>
    {% for _, row in striking_distance.head(15).iterrows() %}
    <tr>
        <td>{{ row.query }}</td>
        <td>{{ "{:.1f}".format(row.avg_position) }}</td>
        <td>{{ "{:,}".format(row.impressions | int) }}</td>
        <td>{{ "{:,}".format(row.clicks | int) }}</td>
        <td>{{ "{:.1%}".format(row.ctr) }}</td>
    </tr>
    {% endfor %}
</table>
{% endif %}

<div class="footer">
    <p>Generated by SquadStack GSC Reporter | Data source: Google Search Console API</p>
    <p>Report covers: {{ period_label }}</p>
</div>

</body>
</html>
"""


def build_report(dry_run=False, monthly=False):
    """Main report builder."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("gsc-report")

    log.info("Authenticating with GSC API...")
    service = get_gsc_service()

    today = datetime.now().date()

    # GSC data has a 2-3 day lag
    end_date = today - timedelta(days=3)

    if monthly:
        current_start = end_date - timedelta(days=29)
        previous_start = current_start - timedelta(days=30)
        previous_end = current_start - timedelta(days=1)
        period_label = f"Monthly: {current_start} to {end_date}"
    else:
        current_start = end_date - timedelta(days=6)
        previous_start = current_start - timedelta(days=7)
        previous_end = current_start - timedelta(days=1)
        period_label = f"Weekly: {current_start} to {end_date}"

    log.info(f"Fetching current period: {current_start} to {end_date}")
    current_df = fetch_gsc_data(
        service, str(current_start), str(end_date),
        dimensions=["query", "page", "date"]
    )

    log.info(f"Fetching previous period: {previous_start} to {previous_end}")
    previous_df = fetch_gsc_data(
        service, str(previous_start), str(previous_end),
        dimensions=["query", "page", "date"]
    )

    log.info(f"Current period: {len(current_df)} rows, Previous: {len(previous_df)} rows")

    # Aggregate by query
    current_by_query = compute_period_metrics(current_df, "query")
    previous_by_query = compute_period_metrics(previous_df, "query")

    # Compute WoW deltas
    with_deltas = compute_deltas(current_by_query, previous_by_query, "query", "WoW")

    # Top movers
    top_gainers = get_top_movers(with_deltas, "clicks", 15, "up")
    top_losers = get_top_movers(with_deltas, "clicks", 15, "down")

    # Cluster analysis
    cluster_summary = compute_cluster_summary(current_by_query, KEYWORD_CLUSTERS)

    # Page performance
    page_perf = compute_page_performance(current_df)
    page_type_summary = pd.DataFrame()
    if not page_perf.empty:
        page_type_summary = page_perf.groupby("page_type").agg(
            pages=("page", "nunique"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_ctr=("avg_ctr", "mean")
        ).reset_index().sort_values("clicks", ascending=False)

    # Branded vs Non-Branded
    brand_keywords = KEYWORD_CLUSTERS.get("Brand", [])
    current_by_query_copy = current_by_query.copy()
    current_by_query_copy["is_branded"] = current_by_query_copy["query"].apply(
        lambda q: any(bk in q.lower() for bk in brand_keywords)
    )
    branded = current_by_query_copy[current_by_query_copy["is_branded"]]
    nonbranded = current_by_query_copy[~current_by_query_copy["is_branded"]]

    # New and lost queries
    current_queries_with_clicks = set(
        current_by_query[current_by_query["clicks"] > 0]["query"]
    )
    previous_queries_with_clicks = set(
        previous_by_query[previous_by_query["clicks"] > 0]["query"]
    )
    new_queries = sorted(current_queries_with_clicks - previous_queries_with_clicks)
    lost_queries = sorted(previous_queries_with_clicks - current_queries_with_clicks)

    # Striking distance: position 4-10 with meaningful impressions
    striking = current_by_query[
        (current_by_query["avg_position"] >= 4)
        & (current_by_query["avg_position"] <= 10)
        & (current_by_query["impressions"] > current_by_query["impressions"].median())
    ].sort_values("impressions", ascending=False)

    # Top-line metrics
    total_clicks = int(current_by_query["clicks"].sum())
    total_impressions = int(current_by_query["impressions"].sum())
    avg_ctr = current_by_query["ctr"].mean() if not current_by_query.empty else 0
    avg_position = current_by_query["avg_position"].mean() if not current_by_query.empty else 0

    prev_clicks = int(previous_by_query["clicks"].sum())
    prev_impressions = int(previous_by_query["impressions"].sum())
    prev_ctr = previous_by_query["ctr"].mean() if not previous_by_query.empty else 0
    prev_position = previous_by_query["avg_position"].mean() if not previous_by_query.empty else 0

    clicks_delta = total_clicks - prev_clicks
    clicks_pct = (clicks_delta / prev_clicks * 100) if prev_clicks > 0 else 0
    imp_delta = total_impressions - prev_impressions
    imp_pct = (imp_delta / prev_impressions * 100) if prev_impressions > 0 else 0
    ctr_delta = avg_ctr - prev_ctr
    pos_delta = prev_position - avg_position  # positive = improvement

    # Generate insights
    report_data = {
        "cluster_summary": cluster_summary,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "current_with_deltas": with_deltas,
        "new_queries": new_queries,
        "lost_queries": lost_queries,
        "page_performance": page_perf
    }
    insights = generate_insights(report_data)
# Convert empty DataFrames to None for Jinja2 template boolean checks
    striking = striking if not striking.empty else None
    cluster_summary = cluster_summary if not cluster_summary.empty else None
    page_type_summary = page_type_summary if not page_type_summary.empty else None
    # Render HTML
    template = Template(EMAIL_TEMPLATE)
    html = template.render(
        period_label=period_label,
        generated_at=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        total_clicks=total_clicks,
        total_impressions=total_impressions,
        avg_ctr=avg_ctr,
        avg_position=avg_position,
        clicks_delta=clicks_delta,
        clicks_pct=clicks_pct,
        imp_delta=imp_delta,
        imp_pct=imp_pct,
        ctr_delta=ctr_delta,
        pos_delta=pos_delta,
        insights=insights,
        branded_clicks=int(branded["clicks"].sum()) if not branded.empty else 0,
        branded_impressions=int(branded["impressions"].sum()) if not branded.empty else 0,
        branded_ctr=branded["ctr"].mean() if not branded.empty else 0,
        branded_position=branded["avg_position"].mean() if not branded.empty else 0,
        nonbranded_clicks=int(nonbranded["clicks"].sum()) if not nonbranded.empty else 0,
        nonbranded_impressions=int(nonbranded["impressions"].sum()) if not nonbranded.empty else 0,
        nonbranded_ctr=nonbranded["ctr"].mean() if not nonbranded.empty else 0,
        nonbranded_position=nonbranded["avg_position"].mean() if not nonbranded.empty else 0,
        cluster_summary=cluster_summary,
        top_gainers=top_gainers,
        top_losers=top_losers,
        page_type_summary=page_type_summary,
        top_pages=page_perf,
        new_queries=new_queries,
      has_new_queries=len(new_queries) > 0,
        striking_distance=striking
    )

    if dry_run:
        output_path = "gsc_report.html"
        with open(output_path, "w") as f:
            f.write(html)
        log.info(f"Report saved to {output_path}")
        print(f"\n{'='*60}")
        print(f"REPORT SUMMARY: {period_label}")
        print(f"{'='*60}")
        print(f"Clicks: {total_clicks:,} ({clicks_delta:+,} WoW)")
        print(f"Impressions: {total_impressions:,} ({imp_delta:+,} WoW)")
        print(f"CTR: {avg_ctr:.1%} ({ctr_delta*100:+.2f}pp WoW)")
        print(f"Position: {avg_position:.1f} ({pos_delta:+.1f} WoW)")
        print(f"\nWins: {len(insights['wins'])}")
        for w in insights["wins"]:
            print(f"  + {w}")
        print(f"\nNeeds Attention: {len(insights['needs_attention'])}")
        for n in insights["needs_attention"]:
            print(f"  ! {n}")
        return

    # Send email
    log.info(f"Sending report to {REPORT_RECIPIENTS}")
    send_email(html, period_label)
    log.info("Report sent successfully.")


def send_email(html_body, period_label):
    """Send the report via SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"GSC Weekly | SquadStack.ai | {period_label}"
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(REPORT_RECIPIENTS)

    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, REPORT_RECIPIENTS, msg.as_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SquadStack GSC Weekly Report")
    parser.add_argument("--dry-run", action="store_true", help="Save HTML locally, don't email")
    parser.add_argument("--monthly", action="store_true", help="Generate monthly report instead of weekly")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks to look back")
    args = parser.parse_args()

    build_report(dry_run=args.dry_run, monthly=args.monthly)
