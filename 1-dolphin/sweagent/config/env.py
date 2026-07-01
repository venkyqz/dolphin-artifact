import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class SWESettings(BaseSettings):
    SWE_HUMAN: bool = False
    # use the interact mode when handling actions, if False, use the parse_actions
    SWE_INTERACT: bool = True
    SWE_LLM_CONFIG: str = ""
    SWE_WORK_DIR: str = ""
    ...

    class Config:
        env_file = '.env'
        extra = 'ignore'

    def require_human_interact(self) -> bool:
        return self.SWE_HUMAN

    def is_interact(self) -> bool:
        return self.SWE_INTERACT

    @property
    def problem(self):
        return os.getenv("PROBLEM_STATEMENT") or ""


settings = SWESettings()

# print(f"SWE_HUMAN: {settings.SWE_HUMAN}")
# print(f"SWE_LLM_CONFIG: {settings.SWE_LLM_CONFIG}")
