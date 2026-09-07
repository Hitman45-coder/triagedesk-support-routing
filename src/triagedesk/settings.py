from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    model_bundle_path: Path = Path("artifacts/releases/demo")
    inference_api_key: str = Field(default="local-dev-key-change-me", min_length=8)
    metrics_token: str = Field(default="local-metrics-token-change-me", min_length=8)
    port: int = Field(default=8000, ge=1, le=65535)
    max_body_bytes: int = Field(default=1_048_576, ge=1024)
    max_text_chars: int = Field(default=2000, ge=3, le=10000)
    max_batch_items: int = Field(default=100, ge=1, le=1000)
    max_in_flight: int = Field(default=2, ge=1, le=32)
    rate_limit_per_minute: int = Field(default=300, ge=1)
    allowed_hosts: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    @property
    def host_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_hosts.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
