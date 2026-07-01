import json
import math
from datetime import datetime

from pydantic import BaseModel, model_validator


class Problem(BaseModel):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str | None
    created_at: datetime
    version: float
    # FAIL_TO_PASS: List[str]
    # PASS_TO_PASS: List[str]
    environment_setup_commit: str

    @model_validator(mode="before")
    def convert_nan_to_empty_string(cls, values):
        value = values.get("hints_text")
        if isinstance(value, float) and math.isnan(value):
            values["hints_text"] = ""
        return values

    @staticmethod
    def load(data):
        # print(data)
        if isinstance(data, str):
            with open(data) as fd:
                data = json.load(fd)

        if isinstance(data, list):
            return [Problem.model_validate(p) for p in data][0]
        else:
            return Problem.model_validate(data)
