import os
import json
import logging
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BondCalculationRequest:
    isin: str
    calculation_type: str  # 'price' or 'yield'
    investment_date: datetime.datetime
    units: int
    input_value: float  # yield_rate for price calculation, price for yield calculation
    bond_data: Dict[str, Any]  # Bond details from the finder

@dataclass
class BondCalculationResponse:
    success: bool
    message: str
    calculation_type: str
    results: Dict[str, Union[float, str, int]]
    bond_details: Dict[str, Any]

class BondCalculatorAgent:
    """
    Agent for calculating bond prices and yields using LLM for validation and processing.
    Integrates with LangChain and Groq for natural language processing.
    """
    
    def __init__(self, 
                 llm_model_name: str = "llama2-70b-4096",
                 api_key: Optional[str] = None,
                 current_date: str = "2025-03-09 21:16:51",
                 current_user: str = "codegeek03"):
        """
        Initialize the Bond Calculator Agent.
        
        Args:
            llm_model_name: Name of the LLM model to use
            api_key: Groq API key (will use environment variable if None)
            current_date: Current date and time in UTC
            current_user: Current user's login
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set and no API key provided.")
        
        # Initialize the LLM client
        self.llm = ChatGroq(model_name=llm_model_name, api_key=api_key)
        
        # Store current date and user
        self.current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
        self.current_user = current_user
        
        # Constants for calculations
        self.days_in_year = 365
        self.payment_frequency = 2  # Semi-annual payments
        
        logger.info(f"BondCalculatorAgent initialized with model: {llm_model_name}")

    def validate_calculation_request(self, request: Dict[str, Any]) -> Tuple[bool, str, Optional[BondCalculationRequest]]:
        """
        Validate the calculation request using LLM for natural language understanding.
        
        Args:
            request: Dictionary containing the calculation request parameters
            
        Returns:
            Tuple[bool, str, Optional[BondCalculationRequest]]: Validation result, error message, and processed request
        """
        system_prompt = """
        You are a bond calculation validator. Verify if the provided calculation request is valid and complete.
        Check for:
        1. Valid ISIN
        2. Valid calculation type (price or yield)
        3. Valid investment date
        4. Valid number of units
        5. Valid input value (yield rate or price)
        Return a JSON response with validation results.
        """

        user_prompt = f"""
        Please validate this bond calculation request:
        {json.dumps(request, default=str, indent=2)}
        
        Current Date: {self.current_date}
        Current User: {self.current_user}
        
        Return JSON format:
        {{
            "is_valid": true/false,
            "error_message": "error details if any",
            "processed_request": {{
                "normalized values if valid"
            }}
        }}
        """

        try:
            # Get validation from LLM
            messages = [
                ("system", system_prompt),
                ("user", user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            validation_result = json.loads(response.content)
            
            if validation_result["is_valid"]:
                # Create BondCalculationRequest from processed data
                processed = validation_result["processed_request"]
                calc_request = BondCalculationRequest(
                    isin=processed["isin"],
                    calculation_type=processed["calculation_type"],
                    investment_date=datetime.datetime.strptime(
                        processed["investment_date"], 
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    units=int(processed["units"]),
                    input_value=float(processed["input_value"]),
                    bond_data=request.get("bond_data", {})
                )
                return True, "", calc_request
            else:
                return False, validation_result["error_message"], None

        except Exception as e:
            logger.error(f"Error in request validation: {str(e)}")
            return False, f"Validation error: {str(e)}", None

    # ... [Rest of the code remains the same] ...

if __name__ == "__main__":
    # Example usage with current date and user
    calculator = BondCalculatorAgent(
        current_date="2025-03-09 21:16:51",
        current_user="codegeek03"
    )
    
    # Example calculation request
    sample_request = {
        "isin": "INE002A08534",
        "calculation_type": "price",
        "investment_date": "2025-03-09 21:16:51",
        "units": 100,
        "input_value": 8.5,  # yield rate for price calculation
        "bond_data": {
            "isin": "INE002A08534",
            "issuer_name": "RELIANCE INDUSTRIES LIMITED",
            "face_value": "1000000",
            "coupon_rate": "9.05%",
            "maturity_date": "17-10-2028"
        }
    }
    
    result = calculator.process_calculation_request(sample_request)
    print(result)