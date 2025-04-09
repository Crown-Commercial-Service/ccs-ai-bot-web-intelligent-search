import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_TITLE = os.getenv("API_TITLE", "Intelligent Search Engine API")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")
    API_DESCRIPTION = os.getenv("API_DESCRIPTION", "API for multi-agent intelligent search engine")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"