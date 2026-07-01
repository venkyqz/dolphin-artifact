import json
import traceback

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    arguments: str
    name: str


class ToolCall(BaseModel):
    index: int
    function: FunctionCall
    id: str
    type: str


class ContentItem(BaseModel):
    type: str
    text: str


class MessageModel(BaseModel):
    content: list[ContentItem] | str
    role: str
    tool_calls: list[ToolCall] | None = None


# ---


class State(BaseModel):
    open_file: str = ""
    working_dir: str = ""


class HistoryItem(BaseModel):
    role: str = ""
    content: str = ""
    agent: str = ""
    tool_calls: list[ToolCall] | None = None


class TrajectoryItem(BaseModel):
    action: str = ""
    observation: str = ""
    response: str = ""
    thought: str = ""
    # state: str


class ModelStats(BaseModel):
    total_cost: float
    instance_cost: float
    tokens_sent: int
    tokens_received: int
    api_calls: int


class Info(BaseModel):
    model_stats: ModelStats
    exit_status: str = ""
    submission: str = ""


class Trajectory(BaseModel):
    agent_name: str = ""
    environment: str = ""
    trajectory: list[TrajectoryItem] = Field(default_factory=list)
    history: list[HistoryItem] = Field(default_factory=list)
    # info: Info = None
    messages: list[MessageModel] = Field(default_factory=list)

    @staticmethod
    def load_dict(data: dict) -> "Trajectory":
        try:
            return Trajectory.model_validate(data)
        except:
            traceback.print_exc()

        try:
            return Trajectory(messages=data)
        except:
            traceback.print_exc()

        return None

    @staticmethod
    def load_trajectory(file_path: str) -> "Trajectory":
        """加载轨迹数据"""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            return Trajectory.model_validate(data)
        except:
            ...
        try:
            return Trajectory(messages=data)
        except:
            ...
