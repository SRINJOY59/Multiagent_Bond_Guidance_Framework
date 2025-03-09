import os
import re
import json
import math
import logging
import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from functools import lru_cache
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
                 llm_model_name: str = "mixtral-8x7b-32768",
                 api_key: Optional[str] = None,
                 current_date: str = "2025-03-09 21:50:36",
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
        
        # Initialize response cache
        self.response_cache = {}
        
        logger.info(f"BondCalculatorAgent initialized with model: {llm_model_name}")

    def get_last_coupon_date(self, current_date: datetime.datetime, maturity_date: datetime.datetime) -> datetime.datetime:
        """Calculate the last coupon date before the current date"""
        days_per_period = 365 // self.payment_frequency
        days_to_maturity = (maturity_date - current_date).days
        periods_to_maturity = math.ceil(days_to_maturity / days_per_period)
        last_coupon = maturity_date - datetime.timedelta(days=periods_to_maturity * days_per_period)
        return last_coupon

    def validate_calculation_request(self, request: Dict[str, Any]) -> Tuple[bool, str, Optional[BondCalculationRequest]]:
        """
        Validate the calculation request using basic validation and LLM.
        
        Args:
            request: Dictionary containing the calculation request parameters
            
        Returns:
            Tuple[bool, str, Optional[BondCalculationRequest]]: Validation result, error message, and processed request
        """
        try:
            # Validate ISIN format
            if not re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', request['isin']):
                return False, "Invalid ISIN format", None

            # Basic field validation
            required_fields = ['isin', 'calculation_type', 'investment_date', 'units', 'input_value', 'bond_data']
            for field in required_fields:
                if field not in request:
                    return False, f"Missing required field: {field}", None

            if request['calculation_type'] not in ['price', 'yield']:
                return False, "Invalid calculation_type. Must be 'price' or 'yield'", None

            try:
                units = int(request['units'])
                if units <= 0:
                    return False, "Units must be positive", None
            except (ValueError, TypeError):
                return False, "Invalid units value", None

            try:
                input_value = float(request['input_value'])
                if input_value <= 0:
                    return False, "Input value must be positive", None
                
                # Validate reasonable ranges
                if request['calculation_type'] == 'yield':
                    if input_value > 10000000:  # Max price validation
                        return False, "Price out of reasonable range", None
                else:  # price calculation
                    if input_value > 100:  # Max yield rate validation
                        return False, "Yield rate out of reasonable range", None
            except (ValueError, TypeError):
                return False, "Invalid input value", None

            # Validate dates
            try:
                investment_date = datetime.datetime.strptime(request['investment_date'], "%Y-%m-%d %H:%M:%S")
                maturity_date = datetime.datetime.strptime(request['bond_data']['maturity_date'], '%d-%m-%Y')
                
                if maturity_date <= investment_date:
                    return False, "Maturity date must be after investment date", None
            except ValueError as e:
                return False, f"Invalid date format: {str(e)}", None

            # Validate bond data
            required_bond_fields = ['isin', 'issuer_name', 'face_value', 'coupon_rate', 'maturity_date']
            for field in required_bond_fields:
                if field not in request['bond_data']:
                    return False, f"Missing required bond data field: {field}", None

            # Create BondCalculationRequest object
            calc_request = BondCalculationRequest(
                isin=request['isin'],
                calculation_type=request['calculation_type'],
                investment_date=investment_date,
                units=units,
                input_value=input_value,
                bond_data=request['bond_data']
            )

            return True, "", calc_request

        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return False, f"Validation error: {str(e)}", None

    def calculate_price(self, request: BondCalculationRequest) -> BondCalculationResponse:
        """Calculate bond price given yield rate"""
        try:
            # Extract and sanitize bond details
            face_value = float(str(request.bond_data['face_value']).replace('₹', '').replace(',', ''))
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
            last_coupon_date = self.get_last_coupon_date(request.investment_date, maturity_date)
            days_since_last_coupon = (request.investment_date - last_coupon_date).days
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
                "investment_date": request.investment_date.strftime('%Y-%m-%d %H:%M:%S'),
                "calculation_date": self.current_date.strftime('%Y-%m-%d %H:%M:%S')
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
        """Calculate yield to maturity given price using binary search method"""
        try:
            # Extract and sanitize bond details
            face_value = float(str(request.bond_data['face_value']).replace('₹', '').replace(',', ''))
            coupon_rate = float(request.bond_data['coupon_rate'].replace('%', '')) / 100
            maturity_date = datetime.datetime.strptime(request.bond_data['maturity_date'], '%d-%m-%Y')
            
            # Calculate time to maturity in years
            time_to_maturity = (maturity_date - request.investment_date).days / self.days_in_year
            
            if time_to_maturity <= 0:
                raise ValueError("Bond has matured")
                
            # Calculate annual coupon payment
            annual_coupon = face_value * coupon_rate
            semi_annual_coupon = annual_coupon / self.payment_frequency
            
            # Calculate accrued interest
            last_coupon_date = self.get_last_coupon_date(request.investment_date, maturity_date)
            days_since_last_coupon = (request.investment_date - last_coupon_date).days
            accrued_interest = (annual_coupon * days_since_last_coupon) / self.days_in_year
            
            # Target dirty price
            dirty_price = request.input_value + accrued_interest
            
            # Binary search parameters
            lower_yield = 0.0001  # 0.01%
            upper_yield = 1.0     # 100%
            tolerance = 0.0001
            max_iterations = 50
            
            def calculate_price_at_yield(ytm):
                """Helper function to calculate price at a given yield"""
                price = 0
                periods = int(time_to_maturity * self.payment_frequency)
                
                for i in range(1, periods + 1):
                    time_to_payment = i / self.payment_frequency
                    discount_factor = 1 / ((1 + ytm) ** time_to_payment)
                    
                    if i == periods:
                        # Last payment includes face value
                        payment = semi_annual_coupon + face_value
                    else:
                        payment = semi_annual_coupon
                        
                    price += payment * discount_factor
                
                return price
                
            # Binary search for yield
            for _ in range(max_iterations):
                current_yield = (lower_yield + upper_yield) / 2
                calculated_price = calculate_price_at_yield(current_yield)
                
                if abs(calculated_price - dirty_price) < tolerance:
                    # Found the yield
                    ytm = current_yield * 100  # Convert to percentage
                    
                    results = {
                        "yield_to_maturity": ytm,
                        "clean_price_per_unit": request.input_value,
                        "clean_price_total": request.input_value * request.units,
                        "dirty_price_per_unit": dirty_price,
                        "dirty_price_total": dirty_price * request.units,
                        "accrued_interest": accrued_interest,
                        "time_to_maturity": time_to_maturity,
                        "annual_coupon": annual_coupon,
                        "units": request.units,
                        "investment_date": request.investment_date.strftime('%Y-%m-%d %H:%M:%S'),
                        "calculation_date": self.current_date.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    return BondCalculationResponse(
                        success=True,
                        message="Yield calculation successful",
                        calculation_type="yield",
                        results=results,
                        bond_details=request.bond_data
                    )
                
                if calculated_price > dirty_price:
                    lower_yield = current_yield
                else:
                    upper_yield = current_yield
                    
            raise ValueError("Yield calculation did not converge within tolerance")
            
        except Exception as e:
            logger.error(f"Error in yield calculation: {str(e)}")
            return BondCalculationResponse(
                success=False,
                message=f"Error calculating yield: {str(e)}",
                calculation_type="yield",
                results={},
                bond_details=request.bond_data
            )

    @lru_cache(maxsize=100)
    def _get_llm_response(self, prompt: str) -> str:
        """Cached LLM response getter"""
        messages = [
            ("system", "You are a bond calculation expert. Format the calculation results into a clear, natural language response."),
            ("user", prompt)
        ]
        return self.llm.invoke(messages).content

    def format_response(self, response: BondCalculationResponse) -> str:
        """Format calculation response using LLM for natural language explanation"""
        try:
            cache_key = f"{response.calculation_type}:{response.bond_details['isin']}:{response.results.get('yield_rate', '')}:{response.results.get('clean_price_per_unit', '')}"
            
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

            prompt = f"""
            Please format these bond calculation results into a clear response:
            {json.dumps(response.__dict__, default=str, indent=2)}
            
            Current Date: {self.current_date}
            Current User: {self.current_user}
            
            Include:
            1. Bond details (ISIN, issuer, etc.)
            2. Calculation type and inputs
            3. Results with explanations
            4. Any relevant notes or warnings
            """
            
            try:
                formatted_response = self._get_llm_response(prompt)
                self.response_cache[cache_key] = formatted_response
                return formatted_response
            except Exception as e:
                logger.warning(f"Error getting LLM response: {str(e)}. Falling back to basic formatting.")
                return self._basic_format_response(response)

        except Exception as e:
            logger.error(f"Error in format_response: {str(e)}")
            return self._basic_format_response(response)

    def _basic_format_response(self, response: BondCalculationResponse) -> str:
        """Basic formatting fallback when LLM is unavailable"""
        if not response.success:
            return f"Error: {response.message}"

        output = []
        output.append(f"\nBond Calculation Results (as of {self.current_date})")
        output.append("=" * 50)
        
        # Bond details
        output.append("Bond Details:")
        output.append(f"ISIN: {response.bond_details.get('isin', 'N/A')}")
        output.append(f"Issuer: {response.bond_details.get('issuer_name', 'N/A')}")
        output.append(f"Face Value: ₹{response.bond_details.get('face_value', 'N/A')}")
        output.append(f"Coupon Rate: {response.bond_details.get('coupon_rate', 'N/A')}")
        output.append(f"Maturity Date: {response.bond_details.get('maturity_date', 'N/A')}")
        
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
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error processing calculation request: {str(e)}")
            return f"Error processing calculation request: {str(e)}"
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Current date and user configuration
    CURRENT_DATE = "2025-03-09 21:21:31"
    CURRENT_USER = "codegeek03"

    try:
        # Initialize the calculator
        calculator = BondCalculatorAgent(
            current_date=CURRENT_DATE,
            current_user=CURRENT_USER
        )
        
        # Example price calculation
        price_request = {
            "isin": "INE002A08534",
            "calculation_type": "price",
            "investment_date": CURRENT_DATE,
            "units": 100,
            "input_value": 8.5,  # yield rate
            "bond_data": {
                "isin": "INE002A08534",
                "issuer_name": "RELIANCE INDUSTRIES LIMITED",
                "face_value": "1000000",
                "coupon_rate": "9.05%",
                "maturity_date": "17-10-2028"
            }
        }
        
        print("\nCalculating Bond Price:")
        print("=" * 50)
        result = calculator.process_calculation_request(price_request)
        print(result)
        
        # Example yield calculation
        yield_request = {
            "isin": "INE002A08534",
            "calculation_type": "yield",
            "investment_date": CURRENT_DATE,
            "units": 100,
            "input_value": 1050000,  # price
            "bond_data": {
                "isin": "INE002A08534",
                "issuer_name": "RELIANCE INDUSTRIES LIMITED",
                "face_value": "1000000",
                "coupon_rate": "9.05%",
                "maturity_date": "17-10-2028"
            }
        }
        
        print("\nCalculating Bond Yield:")
        print("=" * 50)
        result = calculator.process_calculation_request(yield_request)
        print(result)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")