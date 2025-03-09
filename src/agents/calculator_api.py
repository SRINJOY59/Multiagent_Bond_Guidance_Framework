import os
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from bond_calculator import BondCalculatorAgent

# Initialize FastAPI app
app = FastAPI(
    title="Bond Calculator API",
    description="API for calculating bond prices and yields",
    version="1.0.0"
)

# Pydantic models for request/response validation
class BondData(BaseModel):
    isin: str
    issuer_name: str
    face_value: str
    coupon_rate: str
    maturity_date: str

class BondCalculationRequest(BaseModel):
    isin: str = Field(..., description="ISIN of the bond")
    calculation_type: str = Field(..., description="Type of calculation: 'price' or 'yield'")
    investment_date: str = Field(..., description="Investment date in YYYY-MM-DD HH:MM:SS format")
    units: int = Field(..., gt=0, description="Number of bond units")
    input_value: float = Field(..., gt=0, description="Yield rate for price calculation or price for yield calculation")
    bond_data: BondData

# Initialize bond calculator
calculator = BondCalculatorAgent(
    current_date="2025-03-09 21:58:59",
    current_user="codegeek03"
)

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "Bond Calculator API",
        "version": "1.0.0",
        "status": "active",
        "current_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/calculate")
async def calculate_bond(request: BondCalculationRequest):
    """
    Calculate bond price or yield based on the provided parameters
    
    Args:
        request (BondCalculationRequest): The calculation request parameters
        
    Returns:
        dict: Calculation results and formatted response
    """
    try:
        # Convert request to dictionary
        request_dict = request.dict()
        
        # Process calculation request
        result = calculator.process_calculation_request(request_dict)
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)