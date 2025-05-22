from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    """Settings for hosting the server."""

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8000
    LOG_LEVEL: str = "info"


server_settings = ServerSettings()
