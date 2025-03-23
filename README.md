
# Multiagent Bond Guidance Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green)](https://fastapi.tiangolo.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)](https://jupyter.org/)


## Overview

The Multiagent Bond Guidance Framework is an AI-powered layer built for the Tap Bonds platform, designed to enhance bond discovery and research capabilities. This framework integrates multiple specialized agents to provide comprehensive bond analysis and guidance.

## Architecture


<img width="460" alt="image" src="https://github.com/user-attachments/assets/0f3006f9-29ff-4404-8efb-9dc0589f5960" align="center"/>

### 1. Orchestrator Agent (Main Agent)
- Central coordination of user queries
- Intelligent routing to specialized agents
- Response aggregation and formatting

### 2. Specialized Agents

#### Bond Directory Agent
- ISIN-level bond information
- Credit ratings lookup
- Maturity and security type details

#### Bond Finder Agent
- Cross-platform bond comparison
- Yield optimization
- Investment opportunity analysis

#### Cash Flow & Maturity Agent
- Bond cash flow analysis
- Maturity schedule tracking
- Payment timeline management

#### Bond Calculator Agent
- Price calculation from yield rates
- Yield calculation from prices
- Advanced financial metrics computation

### API Implementation

The framework exposes a RESTful API built with FastAPI:

```python
POST /calculate
{
    "isin": "INE002A08534",
    "calculation_type": "price",
    "investment_date": "2025-03-09 21:58:59",
    "units": 100,
    "input_value": 8.5,
    "bond_data": {
        "isin": "INE002A08534",
        "issuer_name": "RELIANCE INDUSTRIES LIMITED",
        "face_value": "1000000",
        "coupon_rate": "9.05%",
        "maturity_date": "17-10-2028"
    }
}
```

## Features

### Bond Calculator
- Price/Yield calculations
- Cash flow analysis
- Accrued interest computation
- Semi-annual coupon handling

### Bond Discovery
- Comprehensive ISIN database
- Multi-platform yield comparison
- Investment opportunity identification

### Financial Analysis
- Issuer creditworthiness assessment
- Financial stability metrics
- Company-level analysis

## Repository Structure

```
├── agents/                 # Agent implementations
│   ├── orchestrator.py    # Main orchestrator agent
│   ├── directory.py       # Bond directory agent
│   ├── finder.py         # Bond finder agent
│   ├── calculator.py     # Bond calculator agent
│   └── cashflow.py       # Cash flow agent
├── api/                   # API implementation
│   └── app.py           # FastAPI application
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Test suite
└── requirements.txt     # Project dependencies
```

## Tech Stack

- **Python**: Core implementation
- **FastAPI**: API framework
- **Jupyter**: Analysis and documentation
- **LangChain**: Agent orchestration
- **Groq**: LLM integration

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/SRINJOY59/Multiagent_Bond_Guidance_Framework.git
cd Multiagent_Bond_Guidance_Framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_api_key"
```

4. Run the API:
```bash
uvicorn api.app:app --reload
```

## Usage Examples

### Bond Price Calculation
```python
from agents.calculator import BondCalculatorAgent

calculator = BondCalculatorAgent()
result = calculator.calculate_price(bond_request)
print(result)
```

### Bond Discovery
```python
from agents.finder import BondFinderAgent

finder = BondFinderAgent()
bonds = finder.search_bonds(criteria)
print(bonds)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TapBonds.com for the platform and data
- Hackathon organizers and mentors
- All contributors and participants
