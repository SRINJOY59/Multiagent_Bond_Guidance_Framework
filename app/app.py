from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

from src.workflow import BondWorkflowChain

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    response: str

workflow_chain = BondWorkflowChain()

@app.post("/process_query", response_model=QueryResponse)
def process_query(request: QueryRequest) -> Any:
    """
    Endpoint to process a bond-related query.

    Args:
        request (QueryRequest): The user's bond-related query

    Returns:
        QueryResponse: The final response from the appropriate agent
    """
    try:
        result = workflow_chain.process_query(request.user_query)
        return QueryResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)