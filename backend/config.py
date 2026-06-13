"""Centralized settings and cached external clients.

Plain os.environ + python-dotenv (the stack the starter already ships, no extra
deps). Clients (LLM + mock API HTTP) are built once and reused.
"""

import os
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.regolo.ai/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("MODEL", "")

MOCK_API_BASE_URL = os.environ.get("MOCK_API_BASE_URL", "https://aldente.yellowtest.it")
MOCK_API_TOKEN = os.environ.get("MOCK_API_TOKEN", "")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "20"))


@lru_cache
def llm() -> OpenAI:
    """OpenAI-compatible client pointed at Regolo.ai or Mistral.

    max_retries=0: retries/backoff are handled in agent._chat so total latency
    stays inside the 30s budget instead of stacking with the SDK's own retries.
    """
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


@lru_cache
def mock_api() -> httpx.Client:
    """HTTP client for the Al Dente mock API, auth header centralized here."""
    return httpx.Client(
        base_url=MOCK_API_BASE_URL,
        headers={"Authorization": f"Bearer {MOCK_API_TOKEN}"},
        timeout=REQUEST_TIMEOUT,
    )
