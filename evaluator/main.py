from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from code_utils import untrusted_check
from typing import Dict, Any


app = FastAPI(title="Code Evaluation Service", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "healthy"}


class EvaluationRequest(BaseModel):
    completion_id: int
    problem: Dict[str, Any]
    solution: str
    identifier: str


@app.post("/evaluate")
async def evaluate_single(task: EvaluationRequest):
    status, details = untrusted_check(task.solution, task.problem["test"])
    return {
        "completion_id": task.completion_id,
        "task_id": task.problem["task_id"],
        "identifier": task.identifier,
        "solution": task.solution,
        "status": status,
        "details": details,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # docker run --name code-evaluator --restart unless-stopped -p 8000:8000 --memory="8g" --cpus="2.0" --memory-swap="12g" -e PYTHONUNBUFFERED=1 code-evaluator
