import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Union
from dateutil.relativedelta import relativedelta

@dataclass
class BondCalculationRequest:
    isin: str
    calculation_type: str  # 'price' or 'yield'
    investment_date: datetime.datetime
    units: int
    input_value: float  # yield_rate for price calculation, price for yield calculation

@dataclass
class BondCalculationResponse:
    success: bool
    message: str
    calculation_type: str
    results: Dict[str, Union[float, str, int]]
    bond_details: Dict[str, str]

class BondDirectory:
    def __init__(self):
        self.bonds = bonds_details_cleaned.csv
            # Add more bonds as neede
    def get_bond_by_isin(self, isin: str) -> Optional[Dict]:
        return self.bonds.get(isin)

    def search_bond_by_issuer(self, issuer_name: str) -> list:
        return [bond for bond in self.bonds.values() 
                if issuer_name.lower() in bond['issuer_name'].lower()]

class BondCalculatorAgent:
    def __init__(self):
        self.bond_directory = BondDirectory()
        self.days_in_year = 365
        self.payment_frequency = 2  # Semi-annual payments

    def _calculate_cashflows(self, bond: Dict, investment_date: datetime.datetime, units: int) -> list:
        """Calculate future cash flows from investment date"""
        face_value = float(bond['face_value'].replace('₹', '').replace(',', '')) * units
        coupon_rate = float(bond['coupon_rate'].replace('%', '')) / 100
        annual_coupon = face_value * coupon_rate
        semi_annual_coupon = annual_coupon / self.payment_frequency

        maturity_date = datetime.datetime.strptime(bond['maturity_date'], '%d-%m-%Y')
        allotment_date = datetime.datetime.strptime(bond['allotment_date'], '%d-%m-%Y')

        cashflows = []
        current_date = allotment_date
        while current_date < investment_date:
            current_date += relativedelta(months=12//self.payment_frequency)

        while current_date <= maturity_date:
            if current_date > investment_date:
                if current_date == maturity_date:
                    cashflows.append((current_date, semi_annual_coupon + face_value))
                else:
                    cashflows.append((current_date, semi_annual_coupon))
            current_date += relativedelta(months=12//self.payment_frequency)

        return cashflows

    def calculate_price(self, request: BondCalculationRequest) -> BondCalculationResponse:
        """Calculate bond price given yield rate"""
        try:
            bond = self.bond_directory.get_bond_by_isin(request.isin)
            if not bond:
                return BondCalculationResponse(
                    success=False,
                    message=f"Bond with ISIN {request.isin} not found",
                    calculation_type="price",
                    results={},
                    bond_details={}
                )

            cashflows = self._calculate_cashflows(bond, request.investment_date, request.units)
            
            # Calculate prices and interest
            dirty_price = 0
            for payment_date, payment_amount in cashflows:
                time_to_payment = (payment_date - request.investment_date).days / self.days_in_year
                discount_factor = 1 / ((1 + request.input_value/100) ** time_to_payment)
                dirty_price += payment_amount * discount_factor

            # Calculate accrued interest
            allotment_date = datetime.datetime.strptime(bond['allotment_date'], '%d-%m-%Y')
            last_coupon_date = allotment_date
            while last_coupon_date + relativedelta(months=6) <= request.investment_date:
                last_coupon_date += relativedelta(months=6)

            days_since_last_coupon = (request.investment_date - last_coupon_date).days
            face_value = float(bond['face_value'].replace('₹', '').replace(',', '')) * request.units
            coupon_rate = float(bond['coupon_rate'].replace('%', '')) / 100
            annual_coupon = face_value * coupon_rate
            daily_interest = annual_coupon / self.days_in_year
            accrued_interest = daily_interest * days_since_last_coupon

            clean_price = dirty_price - accrued_interest
            
            results = {
                "clean_price_per_unit": clean_price / request.units,
                "clean_price_total": clean_price,
                "dirty_price_per_unit": dirty_price / request.units,
                "dirty_price_total": dirty_price,
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
                bond_details=bond
            )

        except Exception as e:
            return BondCalculationResponse(
                success=False,
                message=f"Error calculating price: {str(e)}",
                calculation_type="price",
                results={},
                bond_details={}
            )

    def calculate_yield(self, request: BondCalculationRequest) -> BondCalculationResponse:
        """Calculate yield to maturity given price"""
        try:
            bond = self.bond_directory.get_bond_by_isin(request.isin)
            if not bond:
                return BondCalculationResponse(
                    success=False,
                    message=f"Bond with ISIN {request.isin} not found",
                    calculation_type="yield",
                    results={},
                    bond_details={}
                )

            # Initial yield guess (use coupon rate as starting point)
            yield_guess = float(bond['coupon_rate'].replace('%', ''))
            tolerance = 0.0001
            max_iterations = 100

            # Calculate accrued interest
            allotment_date = datetime.datetime.strptime(bond['allotment_date'], '%d-%m-%Y')
            last_coupon_date = allotment_date
            while last_coupon_date + relativedelta(months=6) <= request.investment_date:
                last_coupon_date += relativedelta(months=6)

            days_since_last_coupon = (request.investment_date - last_coupon_date).days
            face_value = float(bond['face_value'].replace('₹', '').replace(',', '')) * request.units
            coupon_rate = float(bond['coupon_rate'].replace('%', '')) / 100
            annual_coupon = face_value * coupon_rate
            daily_interest = annual_coupon / self.days_in_year
            accrued_interest = daily_interest * days_since_last_coupon

            # Target dirty price
            dirty_price = request.input_value + accrued_interest

            # Newton-Raphson method to find yield
            for _ in range(max_iterations):
                cashflows = self._calculate_cashflows(bond, request.investment_date, request.units)
                price_guess = 0
                for payment_date, payment_amount in cashflows:
                    time_to_payment = (payment_date - request.investment_date).days / self.days_in_year
                    discount_factor = 1 / ((1 + yield_guess/100) ** time_to_payment)
                    price_guess += payment_amount * discount_factor

                if abs(price_guess - dirty_price) < tolerance:
                    results = {
                        "yield_to_maturity": yield_guess,
                        "clean_price_per_unit": request.input_value / request.units,
                        "clean_price_total": request.input_value,
                        "dirty_price_per_unit": dirty_price / request.units,
                        "dirty_price_total": dirty_price,
                        "accrued_interest": accrued_interest,
                        "units": request.units,
                        "investment_date": request.investment_date.strftime('%Y-%m-%d %H:%M:%S')
                    }

                    return BondCalculationResponse(
                        success=True,
                        message="Yield calculation successful",
                        calculation_type="yield",
                        results=results,
                        bond_details=bond
                    )

                # Calculate derivative for Newton-Raphson
                delta = 0.0001
                price_plus_delta = 0
                for payment_date, payment_amount in cashflows:
                    time_to_payment = (payment_date - request.investment_date).days / self.days_in_year
                    discount_factor = 1 / ((1 + (yield_guess + delta)/100) ** time_to_payment)
                    price_plus_delta += payment_amount * discount_factor

                derivative = (price_plus_delta - price_guess) / delta
                yield_guess = yield_guess - (price_guess - dirty_price) / derivative

                if yield_guess < 0:
                    yield_guess = 0

            raise ValueError("Yield calculation did not converge")

        except Exception as e:
            return BondCalculationResponse(
                success=False,
                message=f"Error calculating yield: {str(e)}",
                calculation_type="yield",
                results={},
                bond_details={}
            )

def format_response(response: BondCalculationResponse) -> str:
    """Format the calculation response for the chatbot"""
    if not response.success:
        return f"Error: {response.message}"

    output = []
    output.append("\nBond Details:")
    output.append("=" * 50)
    output.append(f"ISIN: {response.bond_details['isin']}")
    output.append(f"Issuer: {response.bond_details['issuer_name']}")
    output.append(f"Instrument: {response.bond_details['instrument_name']}")
    output.append(f"Credit Rating: {response.bond_details['credit_rating']}")
    output.append("=" * 50)
    
    output.append("\nCalculation Results:")
    output.append("-" * 50)
    results = response.results
    
    if response.calculation_type == "price":
        output.append(f"Yield Rate: {results['yield_rate']:.2f}%")
    else:
        output.append(f"Yield to Maturity: {results['yield_to_maturity']:.2f}%")
        
    output.append(f"Investment Date: {results['investment_date']}")
    output.append(f"Number of Units: {results['units']}")
    output.append(f"\nPer Unit Values:")
    output.append(f"Clean Price: ₹{results['clean_price_per_unit']:,.2f}")
    output.append(f"Dirty Price: ₹{results['dirty_price_per_unit']:,.2f}")
    output.append(f"\nTotal Values:")
    output.append(f"Clean Price: ₹{results['clean_price_total']:,.2f}")
    output.append(f"Dirty Price: ₹{results['dirty_price_total']:,.2f}")
    output.append(f"Accrued Interest: ₹{results['accrued_interest']:,.2f}")
    
    return "\n".join(output)

# Example usage
def process_bond_calculation_request(
    isin: str,
    calculation_type: str,
    investment_date: str,
    units: int,
    input_value: float
) -> str:
    """Process a bond calculation request and return formatted results"""
    
    agent = BondCalculatorAgent()
    
    # Convert investment_date string to datetime
    investment_date = datetime.datetime.strptime(investment_date, '%Y-%m-%d %H:%M:%S')
    
    request = BondCalculationRequest(
        isin=isin,
        calculation_type=calculation_type,
        investment_date=investment_date,
        units=units,
        input_value=input_value
    )
    
    if calculation_type == "price":
        response = agent.calculate_price(request)
    else:
        response = agent.calculate_yield(request)
        
    return format_response(response)