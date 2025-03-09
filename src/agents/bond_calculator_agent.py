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
                 llm_model_name: str = "mixtral-8x7b-32768",
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
        # First, perform basic validation without LLM
        try:
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
            except (ValueError, TypeError):
                return False, "Invalid input value", None

            # Validate investment date format
            try:
                investment_date = datetime.datetime.strptime(request['investment_date'], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return False, "Invalid investment_date format. Use YYYY-MM-DD HH:MM:SS", None

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
            logger.error(f"Error in basic validation: {str(e)}")
            return False, f"Validation error: {str(e)}", None

    def format_validation_prompt(self, request: Dict[str, Any]) -> str:
        """Format the validation prompt for the LLM"""
        return f"""
        Please validate this bond calculation request:
        
        ISIN: {request.get('isin', 'N/A')}
        Calculation Type: {request.get('calculation_type', 'N/A')}
        Investment Date: {request.get('investment_date', 'N/A')}
        Units: {request.get('units', 'N/A')}
        Input Value: {request.get('input_value', 'N/A')}
        
        Bond Details:
        {json.dumps(request.get('bond_data', {}), indent=2)}
        
        Current Date: {self.current_date}
        Current User: {self.current_user}
        
        Please verify:
        1. ISIN format is valid
        2. Calculation type is either 'price' or 'yield'
        3. Investment date is a valid date
        4. Units is a positive integer
        5. Input value is a positive number
        6. Bond data contains all required fields
        
        Respond with 'VALID' or explain why the request is invalid.
        """

    def process_calculation_request(self, query: Dict[str, Any]) -> str:
        """Process a bond calculation request"""
        logger.info("Processing bond calculation request")
        
        try:
            # First perform basic validation
            is_valid, error_message, calculation_request = self.validate_calculation_request(query)
            if not is_valid:
                return f"Invalid calculation request: {error_message}"
            
            # If basic validation passes, perform LLM validation
            prompt = self.format_validation_prompt(query)
            messages = [
                ("system", "You are a bond calculation validator. Verify the request and respond with 'VALID' or explain the issues."),
                ("user", prompt)
            ]
            
            try:
                llm_response = self.llm.invoke(messages)
                validation_text = llm_response.content.strip()
                
                if "VALID" not in validation_text.upper():
                    return f"Invalid calculation request: {validation_text}"
                
            except Exception as e:
                logger.error(f"Error in LLM validation: {str(e)}")
                # If LLM validation fails, proceed with basic validation results
                logger.info("Proceeding with basic validation results only")
            
            # Perform calculation
            if calculation_request.calculation_type == "price":
                response = self.calculate_price(calculation_request)
            else:
                response = self.calculate_yield(calculation_request)
                
            # Format and return response
            return self.format_response(response)
            
        except Exception as e:
            logger.error(f"Error in process_calculation_request: {str(e)}")
            return f"Error processing calculation request: {str(e)}"
        
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
        
        Current Date: {self.current_date}
        Current User: {self.current_user}
        
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
            output.append(f"\nBond Calculation Results (as of {self.current_date})")
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

# ... [Rest of the code remains the same] ...

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