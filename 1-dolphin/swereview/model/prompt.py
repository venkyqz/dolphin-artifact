from pydantic import BaseModel, Field


class CodeElement(BaseModel):
    file: str  # file path
    rel_path: str  # relative path to the module root
    line: int  # line number
    module: str  # module name
    class_name: str  # class name
    function_name: str  # function name


class IssueAnalysisModel(BaseModel):
    reproducible: bool = Field(..., description="Whether the issue is reproducible")
    requires_external_data: bool = Field(..., description="Whether external data is required")
    requires_shell_command: bool = Field(..., description="Whether shell commands need to be executed")
    shell_commands: list[str] | None = Field(None, description="List of shell commands to execute")
    is_code_related: bool = Field(..., description="Whether strongly related to code")
    code_elements: list[CodeElement] | None = Field(
        None,
        description="List of all code elements related to the issue, prioritizing the most relevant ones at the beginning",
    )
    difficulty: str = Field(
        ...,
        description="The difficulty to solve the issue, maybe impossible, hard, medium, simple",
    )


class Config:
    schema_extra = {
        "example": {
            "reproducible": True,
            "requires_external_data": False,
            "requires_shell_command": True,
            "shell_commands": ["ls", "grep 'error'"],
            "is_code_related": True,
            "code_elements": [
                {"category": "class", "name": "MyClass"},
                {"category": "method", "name": "my_method"},
                {"category": "line", "name": "45"},
                {"category": "file", "name": "example.py"},
            ],
        }
    }


def test_model():
    issue_analysis = IssueAnalysisModel(
        reproducible=True,  # 问题可以复现
        requires_external_data=False,  # 问题不需要外部数据
        requires_shell_command=False,  # 问题不涉及shell命令
        shell_commands=None,  # 不需要执行shell命令
        is_code_related=True,  # 问题和代码强相关
        code_elements=[
            CodeElement(category="method", name="pretty_print"),
            CodeElement(category="method", name="sympify"),
            CodeElement(category="class", name="MatrixSymbol"),
            CodeElement(category="file", name="sympy/core/sympify.py"),
            CodeElement(category="file", name="sympy/printing/pretty/pretty.py"),
        ],
    )
    print(issue_analysis)
