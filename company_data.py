from openai import OpenAI
from pydantic import BaseModel

MODEL_NAME = "gpt-4o"


class CompanyDetails(BaseModel):
    name: str
    country: str
    business_type: str
    any_other_details: str


class CompanyNews(BaseModel):
    source_link: str
    news: str
    date: str
    sentiment: str


class CompanyNewsResponse(BaseModel):
    company_news: list[CompanyNews]


def format_prompt_get_company_details(company: str) -> str:
    """Returns the company details prompt"""

    return f"""
You are given a company name and you need to find the key details of the company.

You need to find the key details of the company from the following sources:

Sources:
- LinkedIn
- X (Twitter)
- Company blog
- Press releases

Return the key details in the following format:

json format:
{{"name": "Company Name",
    "country": "Country of Registration",
    "business_type": "Business Type",
    "any_other_details": "Any other details"
}}


<<TARGET_COMPANY>>: {company}
"""


def format_prompt_company_news_prompt(
    company: str, company_details: str, days: int, TRUSTED_NEWS_SOURCES: dict[str, list[str]]
) -> str:
    
    """Returns the company news prompt"""
    gaming_sources = "\n    - ".join(TRUSTED_NEWS_SOURCES["gaming_industry"])
    business_sources = "\n    - ".join(TRUSTED_NEWS_SOURCES["business_sources"])
    company_channels = "\n    - ".join(TRUSTED_NEWS_SOURCES["company_channels"])
    return f"""
    Given the company name company details and the news sources, you need to find all the news for the company in last {
        days
    } days.
        
    You need to find all news from any of the following sources for the company in the last {
        days
    } days:
    Sources:
    gaming industry: {gaming_sources}
    business sources: {business_sources}
    company channels: {company_channels}

    and classify the news into the following categories:
    - Positive
    - Negative
    - Neutral

    Return the news in the following format:
    [
        {{
            "source_link": "https://venturebeat.com/2025/01/01/company-launches-new-product",
            "news": "Summary of the news",
            "date": "2025-01-01",
            "sentiment": "positive"
        }}
    ]
    Prioritise launches, funding, layoffs, hires, licensing, expansion, M&A, KPI milestones, exec thought-leadership, personnel posts.

    <TARGET_COMPANY>: {company}
    <COMPANY_DETAILS>: {company_details}

"""


def get_company_details(client: OpenAI, company: str) -> str:
    response = client.responses.parse(
        model=MODEL_NAME,
        temperature=0.2,
        tools=[{"type": "web_search_preview"}],
        input=format_prompt_get_company_details(company),
        text_format=CompanyDetails,
        tool_choice={"type": "web_search_preview"},
    )
    res: CompanyDetails = response.output_parsed
    print(res)
    return res.model_dump_json(indent=2)


def get_company_news(
    client: OpenAI,
    company: str,
    company_details: str,
    days: int,
    TRUSTED_NEWS_SOURCES: dict[str, list[str]],
) -> str:
    response = client.responses.parse(
        model=MODEL_NAME,
        temperature=0.2,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},
        input=format_prompt_company_news_prompt(
            company, company_details, days, TRUSTED_NEWS_SOURCES
        ),
        text_format=CompanyNewsResponse,
    )
    res: CompanyNewsResponse = response.output_parsed
    print(res)
    return res.model_dump_json(indent=2)
