from urllib.parse import urlparse
from typing import Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
import time
import random
from fake_useragent import UserAgent


class Crawler:
    def __init__(
        self,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        rotate_user_agents: bool = True,
        respect_robots_txt: bool = True,
    ):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.last_request_time: Dict[str, float] = {}
        self.session = self._create_session()
        self.rotate_user_agents = rotate_user_agents
        self.user_agent: Optional[UserAgent] = (
            UserAgent() if rotate_user_agents else None
        )
        self.respect_robots_txt = respect_robots_txt

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _wait_for_rate_limit(self, domain: str) -> None:
        current_time = time.time()
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed + random.uniform(0, 0.1))
        self.last_request_time[domain] = time.time()
        return None

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a GET request with rate limiting and retries."""
        domain = urlparse(url).netloc
        self._wait_for_rate_limit(domain)

        headers = kwargs.pop("headers", {})
        if self.rotate_user_agents and self.user_agent is not None:
            headers["User-Agent"] = self.user_agent.random

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, headers=headers, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2**attempt)
                    time.sleep(sleep_time)
                    continue
        raise last_error  # type: ignore[misc]

    def head(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a HEAD request with rate limiting and retries."""
        domain = urlparse(url).netloc
        self._wait_for_rate_limit(domain)

        headers = kwargs.pop("headers", {})
        if self.rotate_user_agents and self.user_agent is not None:
            headers["User-Agent"] = self.user_agent.random

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.head(url, headers=headers, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2**attempt)
                    time.sleep(sleep_time)
                    continue
        raise last_error  # type: ignore[misc]
