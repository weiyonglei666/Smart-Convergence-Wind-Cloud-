from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Resolve the project env file from this module instead of the process CWD.
    # The scheduler is also launched by IDEs/Task Scheduler, where CWD may not
    # be ``backend``; a relative env_file can silently load no NWP settings.
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
    )

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = ""
    forecast_table_prefix: str = ""
    nwp_ssh_host: str = "nwp-host"
    nwp_ssh_port: int = 22
    nwp_ssh_user: str = ""
    nwp_remote_dir: str = "/path/to/nwp/48h"
    nwp_refresh_hours: str = "2,4,9,14,16,21"
    timezone_name: str = "Asia/Shanghai"
    realtime_interval_seconds: int = 30
    admin_username: str = "admin"
    admin_password: str = "change-me"

@lru_cache
def get_settings() -> Settings:
    return Settings()
