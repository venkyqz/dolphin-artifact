from pydantic import BaseModel


class ToolRequest(BaseModel):
    workspace: str
    command: str

class ToolResponse(BaseModel):
    status: str
    command: str
    message: str
    completion: bool
    submission: str | None

    def format_str(self):
        # FIXME: DO NOT use this api, since we should keep empty message for post prompt process
        return (
            f"[Status]\n{self.status}\n[Command]\n{self.command}\n[Output]\n{self.message}"
        )
