"""
Streamlit News Agent for Mobile Gaming Companies
================================================
A lightweight UI that lets users:
1. Enter a company/app name and fetch a structured news summary using OpenAI GPT‑4o + openai‑search.
2. Add that company to a persistent watch‑list.
3. Auto‑refresh every Monday 09:00 IST via APScheduler.

Environment
-----------
- streamlit>=1.33
- openai>=1.30  # GPT‑4o + openai‑search support
- apscheduler
- python‑dotenv (optional)

```bash
pip install streamlit openai apscheduler python-dotenv
```

Set `OPENAI_API_KEY` in your environment or a `.env` file.
"""

from __future__ import annotations

import os
import json
import datetime as dt
from pathlib import Path

import streamlit as st
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler

from company_data import CompanyDetails, get_company_details, get_company_news
from company_personnel import get_company_personnel, get_company_personnel_news

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
WATCHLIST_FILE = Path("watchlist.json")
COMPANY_DATA_FILE = Path("company_data.json")
NUMBER_OF_DAYS = 180
REFRESH_DAY = "mon"  # Weekly refresh every Monday
REFRESH_HOUR = 9  # 09:00 IST
MODEL_NAME = "gpt-4.1"  # Use "gpt-4o" if you have access

# Configuration for news sources
TRUSTED_NEWS_SOURCES = {
    "gaming_industry": [
        "Pocket Gamer",
        "PocketGamer.biz",
        "Gamezebo",
        "Deconstructor of Fun",
        "IGN",
        "Gamespot",
        "Kotaku",
        "VentureBeat (GamesBeat)",
    ],
    "business_sources": ["TechCrunch", "Bloomberg", "Reuters"],
    "company_channels": [
        "LinkedIn posts",
        "X (Twitter) posts",
        "Company blog",
        "Press releases",
    ],
}

# --------------------------------------------------
# OpenAI client
# --------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------
# Prompt builder
# --------------------------------------------------


def format_prompt(
    company: str,
    personnel: str,
    personnel_news: str,
    company_details: str,
    company_news: str,
) -> str:
    """Returns the system prompt that instructs GPT‑4o to act as the News Agent."""

    return f"""
🧠 Prompt for a News AI Agent (Gaming-Focused)
You are a News Intelligence Agent trained to monitor and summarize financial-impacting developments for mobile gaming companies and apps. 
I will give you all the relevant information about the company and the key personnel and their news. I want you to format the information. Add comments if needed.

FORMAT THE GIVEN INFORMATION IN A READABLE FORMAT.

<TARGET_COMPANY>: {company}
<PERSONNEL>: {personnel}
<PERSONNEL_NEWS>: {personnel_news}
<COMPANY_DETAILS>: {company_details}
<COMPANY_NEWS>: {company_news}

**Output (nothing else):**

### Company Details
- Details of the company

### Key Personnel
- Details of the key personnel

### 📰 Company News (Past 6 Months) [Ensure that the source link is mentioned in the citation]
| Sentiment(✅/Neutral/❌) | Source | Headline | Summary (≤ 30 words) | [Citation-Mention the source link] |
|------|--------|----------|----------------------|------------|
| ...... |

### 📰 Key Personnel News (Past 6 Months) [Ensure that the source link is mentioned in the citation]
| Sentiment(✅/Neutral/❌) | Source | Headline | Summary (≤ 30 words) | [Citation-Mention the source link] |
|------|--------|----------|----------------------|------------|
| ...... |

### ⚡ Financial & Strategic Sentiment
➡️ Positive / Negative / Neutral — *one-sentence justification.*

*Do not output any content outside the sections above.*

"""


# --------------------------------------------------
# Persistence helpers
# --------------------------------------------------


def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_FILE.read_text()) if WATCHLIST_FILE.exists() else []
    except Exception:
        return []


def save_watchlist(watchlist: list[str]):
    WATCHLIST_FILE.write_text(json.dumps(sorted(set(watchlist))))


def load_company_data() -> dict:
    """Load company data from JSON file."""
    try:
        return (
            json.loads(COMPANY_DATA_FILE.read_text())
            if COMPANY_DATA_FILE.exists()
            else {}
        )
    except Exception:
        return {}


def save_company_data(company: str, data: dict):
    """Save company data to JSON file."""
    all_data = load_company_data()
    all_data[company]["final_summary"] = data
    COMPANY_DATA_FILE.write_text(json.dumps(all_data, indent=2))


def get_company_data(company: str) -> dict | None:
    """Get data for a specific company."""
    return load_company_data().get(company, {}).get("final_summary")


# --------------------------------------------------
# GPT call wrapper
# --------------------------------------------------


def save_intermediate_data(company: str, key: str, data: str):
    all_data = load_company_data()
    if company not in all_data:
        all_data[company] = {}
    all_data[company][key] = json.loads(data)
    COMPANY_DATA_FILE.write_text(json.dumps(all_data, indent=2))


def run_news_agent(company: str) -> str:
    """Calls OpenAI with web search tool to obtain a summary."""

    personnel = get_company_personnel(client, company)
    save_intermediate_data(company, "personnel", personnel)

    personnel_news = get_company_personnel_news(
        client, company, personnel, NUMBER_OF_DAYS, TRUSTED_NEWS_SOURCES
    )
    save_intermediate_data(company, "personnel_news", personnel_news)

    company_details = get_company_details(client, company)
    save_intermediate_data(company, "company_details", company_details)

    company_news = get_company_news(
        client, company, company_details, NUMBER_OF_DAYS, TRUSTED_NEWS_SOURCES
    )
    # save_intermediate_data(company, "company_news", company_news)
    # personnel = load_company_data()[company]["personnel"]
    # personnel_news = load_company_data()[company]["personnel_news"]
    # company_details = load_company_data()[company]["company_details"]
    # company_news = load_company_data()[company]["company_news"]
    print("--------------------------------")
    print(
        format_prompt(company, personnel, personnel_news, company_details, company_news)
    )
    print("--------------------------------")


    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": format_prompt(company, personnel, personnel_news, company_details, company_news)},
        ],
    )
    return response.choices[0].message.content


# --------------------------------------------------
# Background scheduler
# --------------------------------------------------


def schedule_weekly_refresh(callback, watchlist: list[str]):
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    def refresh_all():
        for comp in watchlist:
            try:
                summary = callback(comp)
                save_company_data(
                    comp,
                    {
                        "summary": summary,
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                )
            except Exception as exc:
                print(f"[Scheduler] Error processing {comp}: {exc}")

    scheduler.add_job(
        refresh_all,
        "cron",
        day_of_week=REFRESH_DAY,
        hour=REFRESH_HOUR,
        minute=0,
    )
    scheduler.start()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="🎮 Gaming News Agent", layout="wide")

watchlist = load_watchlist()

st.title("🎮 Gaming News Intelligence Agent")

company_input = st.text_input("Enter a gaming company or app name:").strip()
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Fetch Summary", disabled=not company_input):
        with st.spinner("Fetching latest news..."):
            try:
                summary = run_news_agent(company_input)
                save_company_data(
                    company_input,
                    {
                        "summary": summary,
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                )
            except Exception as e:
                st.error(f"Error fetching data: {e}")

with col2:
    if st.button("Add to Watch‑List", disabled=not company_input):
        if company_input not in watchlist:
            watchlist.append(company_input)
            save_watchlist(watchlist)
            st.success(f"{company_input} added to watch‑list!")
        else:
            st.info("Already in watch‑list.")

st.markdown("---")

# Display helpers


def display_summary(comp: str):
    data = get_company_data(comp)
    if not data:
        st.info("No cached data – click *Fetch Summary* first.")
        return
    st.subheader(comp)
    st.caption(f"Last updated: {data['timestamp']}")
    st.markdown(data["summary"], unsafe_allow_html=True)


current, watch = st.tabs(["🔍 Current Result", "📋 Watch‑List"])

with current:
    if company_input:
        display_summary(company_input)

with watch:
    if not watchlist:
        st.info("Watch‑list is empty. Add companies using the input above.")
    for comp in watchlist:
        with st.expander(comp, expanded=False):
            display_summary(comp)

# Start scheduler once per app session
if "_scheduler_started" not in st.session_state:
    schedule_weekly_refresh(run_news_agent, watchlist)
    st.session_state._scheduler_started = True
