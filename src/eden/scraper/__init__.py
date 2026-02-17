"""Eden web scraper for RHS plant and gardening data."""

BASE_URL = "https://www.rhs.org.uk"
ADVICE_SEARCH_API = f"{BASE_URL}/api/advice/Search"
PLANTS_API_BASE = "https://lwapp-uks-prod-psearch-01.azurewebsites.net"
DETAIL_ENDPOINT = f"{PLANTS_API_BASE}/api/v1/plants/details"
USER_AGENT = "Eden/0.1 (RHS plant data research project)"

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between requests
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds
MAX_CONCURRENT = 5  # async concurrency limit
