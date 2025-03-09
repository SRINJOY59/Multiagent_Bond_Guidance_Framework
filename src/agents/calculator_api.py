from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import uvicorn
from bond_calculator import BondCalculatorAgent

app = FastAPI(
    title="Bond Calculator API",
    description="API for calculating bond prices and yields using LLM-powered validation",
    version="1.0.0"
)

class BondRequest(BaseModel):
    isin: str
    calculation_type: str
    investment_date: str
    units: int
    input_value: float
    bond_data: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "Bond Calculator API",
        "version": "1.0.0",
        "endpoints": [
            "/calculate",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/calculate")
async def calculate_bond(request: BondRequest):
    try:
        # Initialize calculator with current timestamp
        calculator = BondCalculatorAgent(
            current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            current_user="api_user"  # You might want to get this from auth
        )
        
        # Convert Pydantic model to dict and process
        calculation_result = calculator.process_calculation_request(request.dict())
        
        return {
            "status": "success",
            "result": calculation_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Calculation failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("calculator_api:app", host="0.0.0.0", port=8000, reload=True)