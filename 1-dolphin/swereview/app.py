import json

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from path import Path

from swereview.analysis.analysis_problem import search_with_id
from swereview.model import Problem, Trajectory

app = FastAPI()

Path("static").mkdir_p()
Path("templates").mkdir_p()

# 设置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 存储已加载的轨迹数据
trajectories: dict[str, Trajectory] = {}

# problems: Dict[str, Problem] = {}

swe_bench_problem = [
    # "swe-bench.dev.csv",
    # "swe-bench.test.csv",
    "swe-bench-lite.test.csv",
    # "swe-bench-verify.csv",
]

df_list = [pd.read_csv(Path("data").joinpath(i)) for i in swe_bench_problem]


# Add this function to initialize problems from CSV files
def initialize_problems():
    problems_dict = {}
    for df in df_list:
        records = df.to_dict(orient="records")
        for record in records:
            problem = Problem.model_validate(record)
            problems_dict[problem.instance_id] = problem
    return problems_dict


# Modify the global problems dictionary initialization
problems: dict[str, Problem] = initialize_problems()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "trajectories": trajectories, "problems": problems},
    )


# @app.get("/", response_class=HTMLResponse)
# async def index():
#     # Return static index.html page directly
#     with open("static/index.html", "r") as f:
#         return HTMLResponse(content=f.read())


@app.get("/info")
async def get_info():
    """Get basic information about trajectories and problems"""
    return {
        "trajectories": trajectories,
        "problems": problems,
    }


@app.get("/search/problem")
async def search_problem(request: Request, query: str):
    for df in df_list:
        df_filtered = search_with_id(query, df)
        if len(df_filtered) > 0:
            records = df_filtered.to_dict(
                orient="records",
            )

            items = [Problem.model_validate(r) for r in records]
            for it in items:
                problems[it.instance_id] = it

            return templates.TemplateResponse(
                "problem.html",
                {
                    "request": request,
                    "problem": items[0],
                    "file_name": items[0].instance_id,
                },
            )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "trajectories": trajectories,
            "problems": problems,
            "message": "No problem found for the given query",
        },
    )


@app.post("/upload_trajectory")
async def upload_traj(file: UploadFile):
    content = await file.read()
    data = json.loads(content)
    trajectory = Trajectory.load_dict(data)
    file_name = file.filename
    key = file_name.split(".")[0]
    trajectories[key] = trajectory
    return {"status": "success", "file_name": file_name}


@app.post("/upload_problem")
async def upload_problem(file: UploadFile):
    content = await file.read()
    data = json.loads(content)
    problem = Problem.load(data)
    file_name = file.filename
    problems[file_name] = problem
    return {"status": "success", "file_name": file_name}


@app.get("/trajectory/{file_name}")
async def get_trajectory(request: Request, file_name: str):
    key = file_name.split(".")[0]
    trajectory = trajectories.get(key)
    if not trajectory:
        return Response(content="Trajectory " + key + " not found", status_code=204)
    # Return rendered HTML template
    return templates.TemplateResponse(
        "trajectory_content.html",
        {"request": request, "trajectory": trajectory, "file_name": key},
    )


@app.get("/problem/{file_name}")
async def get_problem(request: Request, file_name: str):
    key = file_name.split(".")[0]
    problem = problems.get(key)
    if not problem:
        return Response(content="Problem " + key + " not found", status_code=204)
    # Return rendered HTML template
    return templates.TemplateResponse(
        "problem_content.html",
        {"request": request, "problem": problem, "file_name": key},
    )


@app.get("/get_lists")
async def get_lists():
    return {
        "trajectories": list(trajectories.keys()),
        "problems": list(problems.keys()),
    }


if __name__ == "__main__":
    # while run with variable, reload must be False
    uvicorn.run(app, host="0.0.0.0", port=8090, reload=False)
