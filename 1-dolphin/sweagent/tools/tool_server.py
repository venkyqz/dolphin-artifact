from fastapi import FastAPI, HTTPException

from sweagent.agent.types import ToolRequest, ToolResponse
from sweagent.tools.query import execute_command
from sweagent.utils.telemetry import get_format_logger

log = get_format_logger("ToolServer")

# Server
app = FastAPI()


@app.post("/api/tool", response_model=ToolResponse)
async def call_tool(request: ToolRequest):
    try:
        # Process the text input
        output = execute_command(request.command,root_path= request.workspace)
        return {"status": "success", "message": f"{output}"}
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sweagent.agent.tool_server:app", host="0.0.0.0", port=8010, reload=True
    )
