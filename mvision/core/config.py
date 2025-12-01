from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str  # lowercase matches the attribute we want

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()
