import os
import json
import logging
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from decimal import Decimal, ROUND_HALF_UP

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
                 api_key: Optional[str] = None):
        """
        Initialize the Bond Calculator Agent.
        
        Args:
            llm_model_name: Name of the LLM model to use
            api_key: Groq API key (will use environment variable if None)
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
        
        # Constants for calculations
        self.days_in_year = 365
        self.payment_frequency = 2  # Semi-annual payments
        
        logger.info(f"BondCalculatorAgent initialized with model: {llm_model_name}")

    def validate_calculation_request(self, request: Dict[str, Any]) -> tuple[bool, str, Optional[BondCalculationRequest]]:
        """
        Validate the calculation request using LLM for natural language understanding.
        
        Args:
            request: Dictionary containing the calculation request parameters
            
        Returns:
            tuple: (is_valid, error_message, processed_request)
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

    def calculate_price(self, request: BondCalculationRequest) -> BondCalculationResponse:
        """Calculate bond price given yield rate"""
        try:
            # Extract bond details
            face_value = float(request.bond_data['face_value'].replace('₹', '').replace(',', ''))
            coupon_rate = float(request.bond_data['coupon_rate'].replace('%', '')) / 100
            maturity_date = datetime.datetime.strptime(request.bond_data['maturity_date'], '%d-%m-%Y')
            
            # Calculate cash flows
            annual_coupon = face_value * coupon_rate
            semi_annual_coupon = annual_coupon / self.payment_frequency
            
            # Generate future cash flows
            cashflows = []
            current_date = request.investment_date
            while current_date <= maturity_date:
                if current_date > request.investment_date:
                    if current_date == maturity_date:
                        cashflows.append((current_date, semi_annual_coupon + face_value))
                    else:
                        cashflows.append((current_date, semi_annual_coupon))
                current_date += datetime.timedelta(days=365//self.payment_frequency)

            # Calculate present value
            dirty_price = 0
            yield_rate = request.input_value / 100
            
            for payment_date, amount in cashflows:
                time_to_payment = (payment_date - request.investment_date).days / self.days_in_year
                discount_factor = 1 / ((1 + yield_rate) ** time_to_payment)
                dirty_price += amount * discount_factor

            # Calculate accrued interest
            days_since_last_coupon = (request.investment_date - maturity_date).days % (365//self.payment_frequency)
            accrued_interest = (annual_coupon * days_since_last_coupon) / self.days_in_year
            
            clean_price = dirty_price - accrued_interest
            
            results = {
                "clean_price_per_unit": clean_price,
                "clean_price_total": clean_price * request.units,
                "dirty_price_per_unit": dirty_price,
                "dirty_price_total": dirty_price * request.units,
                "accrued_interest": accrued_interest,
                "yield_rate": request.input_value,
                "units": request.units,
                "investment_date": request.investment_date.strftime('%Y-%m-%d %H:%M:%S')
            }

            return BondCalculationResponse(
                success=True,
                message="Price calculation successful",
                calculation_type="price",
                results=results,
                bond_details=request.bond_data
            )

        except Exception as e:
            logger.error(f"Error in price calculation: {str(e)}")
            return BondCalculationResponse(
                success=False,
                message=f"Error calculating price: {str(e)}",
                calculation_type="price",
                results={},
                bond_details=request.bond_data
            )

    def calculate_yield(self, request: BondCalculationRequest) -> BondCalculationResponse:
        """Calculate yield to maturity given price"""
        try:
            # Extract bond details
            face_value = float(request.bond_data['face_value'].replace('₹', '').replace(',', ''))
            coupon_rate = float(request.bond_data['coupon_rate'].replace('%', '')) / 100
            maturity_date = datetime.datetime.strptime(request.bond_data['maturity_date'], '%d-%m-%Y')
            
            # Initial yield guess (use coupon rate as starting point)
            yield_guess = coupon_rate * 100
            tolerance = 0.0001
            max_iterations = 100
            
            # Calculate accrued interest
            days_since_last_coupon = (request.investment_date - maturity_date).days % (365//self.payment_frequency)
            annual_coupon = face_value * coupon_rate
            accrued_interest = (annual_coupon * days_since_last_coupon) / self.days_in_year
            
            # Target dirty price
            dirty_price = request.input_value + accrued_interest
            
            # Newton-Raphson method to find yield
            for _ in range(max_iterations):
                # Calculate price at current yield guess
                price_calc_request = BondCalculationRequest(
                    isin=request.isin,
                    calculation_type="price",
                    investment_date=request.investment_date,
                    units=request.units,
                    input_value=yield_guess,
                    bond_data=request.bond_data
                )
                
                price_result = self.calculate_price(price_calc_request)
                if not price_result.success:
                    raise ValueError(price_result.message)
                
                price_guess = price_result.results["dirty_price_per_unit"]
                
                if abs(price_guess - dirty_price) < tolerance:
                    results = {
                        "yield_to_maturity": yield_guess,
                        "clean_price_per_unit": request.input_value,
                        "clean_price_total": request.input_value * request.units,
                        "dirty_price_per_unit": dirty_price,
                        "dirty_price_total": dirty_price * request.units,
                        "accrued_interest": accrued_interest,
                        "units": request.units,
                        "investment_date": request.investment_date.strftime('%Y-%m-%d %H:%M:%S')
                    }

                    return BondCalculationResponse(
                        success=True,
                        message="Yield calculation successful",
                        calculation_type="yield",
                        results=results,
                        bond_details=request.bond_data
                    )

                # Calculate derivative for Newton-Raphson
                delta = 0.0001
                price_calc_request.input_value = yield_guess + delta
                price_plus_delta = self.calculate_price(price_calc_request).results["dirty_price_per_unit"]
                
                derivative = (price_plus_delta - price_guess) / delta
                yield_guess = yield_guess - (price_guess - dirty_price) / derivative
                
                if yield_guess < 0:
                    yield_guess = 0

            raise ValueError("Yield calculation did not converge")

        except Exception as e:
            logger.error(f"Error in yield calculation: {str(e)}")
            return BondCalculationResponse(
                success=False,
                message=f"Error calculating yield: {str(e)}",
                calculation_type="yield",
                results={},
                bond_details=request.bond_data
            )

    def format_response(self, response: BondCalculationResponse) -> str:
        """Format calculation response using LLM for natural language explanation"""
        system_prompt = """
        You are a bond calculation expert. Format the calculation results into a clear,
        natural language response. Include all relevant details and explain the results
        in a way that's easy to understand.
        """

        user_prompt = f"""
        Please format these bond calculation results into a clear response:
        {json.dumps(response.__dict__, default=str, indent=2)}
        
        Include:
        1. Bond details (ISIN, issuer, etc.)
        2. Calculation type and inputs
        3. Results with explanations
        4. Any relevant notes or warnings
        """

        try:
            messages = [
                ("system", system_prompt),
                ("user", user_prompt)
            ]
            
            llm_response = self.llm.invoke(messages)
            return llm_response.content

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            # Fallback to basic formatting
            if not response.success:
                return f"Error: {response.message}"

            output = []
            output.append("\nBond Calculation Results:")
            output.append("=" * 50)
            
            # Bond details
            output.append("Bond Details:")
            output.append(f"ISIN: {response.bond_details.get('isin', 'N/A')}")
            output.append(f"Issuer: {response.bond_details.get('issuer_name', 'N/A')}")
            
            # Calculation results
            output.append("\nCalculation Results:")
            output.append(f"Type: {response.calculation_type.title()}")
            
            results = response.results
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if 'price' in key:
                        output.append(f"{key.replace('_', ' ').title()}: ₹{value:,.2f}")
                    elif 'rate' in key or 'yield' in key:
                        output.append(f"{key.replace('_', ' ').title()}: {value:.2f}%")
                    else:
                        output.append(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    output.append(f"{key.replace('_', ' ').title()}: {value}")

            return "\n".join(output)

    def process_calculation_request(self, query: Dict[str, Any]) -> str:
        """
        Process a bond calculation request from the orchestrator.
        
        Args:
            query: Dictionary containing the calculation request
            
        Returns:
            str: Formatted calculation results
        """
        logger.info("Processing bond calculation request")
        
        # Validate the request
        is_valid, error_message, calculation_request = self.validate_calculation_request(query)
        
        if not is_valid:
            return f"Invalid calculation request: {error_message}"
            
        # Perform calculation
        if calculation_request.calculation_type == "price":
            response = self.calculate_price(calculation_request)
        else:
            response = self.calculate_yield(calculation_request)
            
        # Format and return response
        return self.format_response(response)

if __name__ == "__main__":
    # Example usage
    calculator = BondCalculatorAgent()
    
    # Example calculation request
    sample_request = {
        "isin": "INE002A08534",
        "calculation_type": "price",
        "investment_date": "2025-03-09 21:13:46",
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