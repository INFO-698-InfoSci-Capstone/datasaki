from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SMTP_SERVER: str
    SMTP_PORT: int
    EMAIL_USER: str
    EMAIL_PASSWORD: str
    SECRET_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
