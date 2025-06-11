from openai import OpenAI
from pydantic import BaseModel

MODEL_NAME = "gpt-4o"


class CompanyPersonnel(BaseModel):
    name: str
    role: str
    linkedin: str
    x: str


class CompanyPersonnelResponse(BaseModel):
    company_personnel: list[CompanyPersonnel]


class CompanyPersonnelNews(BaseModel):
    source_link: str
    news: str
    date: str
    sentiment: str


class CompanyPersonnelNewsResponse(BaseModel):
    company_personnel_news: list[CompanyPersonnelNews]


def format_prompt_get_company_personnel(company: str) -> str:
    """Returns the company personnel prompt"""

    return f"""
You are expert in finding the key personnel of a company.

You are given a company name and you need to find the key personnel of the company.

You need to find the key personnel of the company from the following sources:

Sources:
- LinkedIn
- X (Twitter)
- Company blog
- Press releases

Return the key personnel in the following format:

json format (leave blank if you don't find the information):
[
  {{"name": "John Doe",
    "role": "CEO",
    "linkedin": "https://www.linkedin.com/in/john-doe",
    "x": "https://x.com/john-doe"
  }}
]

<<TARGET_COMPANY>>: {company}
"""


def format_prompt_company_personnel_news_prompt(
    company: str, personnel: str, days: int, TRUSTED_NEWS_SOURCES: dict[str, list[str]]
) -> str:
    """Returns the company personnel news prompt"""
    gaming_sources = "\n    - ".join(TRUSTED_NEWS_SOURCES["gaming_industry"])
    business_sources = "\n    - ".join(TRUSTED_NEWS_SOURCES["business_sources"])
    company_channels = "\n    - ".join(TRUSTED_NEWS_SOURCES["company_channels"])
    print(f"gaming_sources: {gaming_sources}")
    return f"""
    Given the company name and the key personnel of the company, you need to find all the news for each of the key personnel in last {days} days.
    
    You need to find the news for each of the key personnel from the following sources:
    Sources:
    gaming industry: {gaming_sources}
    business sources: {business_sources}
    company channels: {company_channels}

    For each news you find, classify the news into the following categories:
    - Positive
    - Negative
    - Neutral

    Return the news in the following format:
    [
        {{
            "source_link": "https://www.linkedin.com/in/john-doe",
            "news": "Summary of the news",
            "date": "2025-01-01",
            "sentiment": "positive",
        }}
    ]
    Prioritise launches, funding, layoffs, hires, licensing, expansion, M&A, KPI milestones, exec thought-leadership, personnel posts.
    You must atleast 5 news in total.
    <TARGET_COMPANY>: {company}
    <KEY_PERSONNEL>: {personnel}

"""


def get_company_personnel_news(
    client: OpenAI,
    company: str,
    personnel: str,
    days: int,
    TRUSTED_NEWS_SOURCES: dict[str, list[str]],
) -> str:
    response = client.responses.parse(
        model=MODEL_NAME,
        temperature=0.2,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},
        input=format_prompt_company_personnel_news_prompt(
            company, personnel, days, TRUSTED_NEWS_SOURCES
        ),
        text_format=CompanyPersonnelNewsResponse,
    )
    res: CompanyPersonnelNewsResponse = response.output_parsed
    print(res)
    print(response)
    return res.model_dump_json(indent=2)


def get_company_personnel(client: OpenAI, company: str) -> str:
    response = client.responses.parse(
        model=MODEL_NAME,
        temperature=0.2,
        tools=[{"type": "web_search_preview"}],
        input=format_prompt_get_company_personnel(company),
        text_format=CompanyPersonnelResponse,
        tool_choice={"type": "web_search_preview"},
    )
    res: CompanyPersonnelResponse = response.output_parsed
    print(res)
    return res.model_dump_json(indent=2)
