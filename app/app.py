import os
import sys
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from typing import Dict, Any, Optional


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.workflow import BondWorkflowChain
from src.agents.bond_calculator_agent import BondCalculatorAgent
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    response: str
class BondRequest(BaseModel):
    isin: str
    calculation_type: str
    investment_date: str
    units: int
    input_value: float
    bond_data: Dict[str, Any]

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
    
@app.post("/calculate")
async def calculate_bond(query : str):
    try:
        calculator = BondCalculatorAgent(
            current_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            current_user="api_user"  
        )
        
        calculation_result = calculator.process_query(query)
        
        return {
            "status": "success",
            "result": calculation_result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Calculation failed: {str(e)}"
        )
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)