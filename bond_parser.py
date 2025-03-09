from datetime import datetime
import json
from typing import Dict, Any
import pandas as pd

class BondDataParser:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        
    def parse_non_null_fields(self) -> Dict[str, Any]:
        """Extract all non-null fields from the bond data"""
        parsed_data = {
            "basic_info": {
                "id": self.data["id"],
                "isin": self.data["isin"],
                "company_name": self.data["company_name"],
                "issue_size": self.data["issue_size"],
                "allotment_date": self.data["allotment_date"],
                "maturity_date": self.data["maturity_date"],
                "created_at": self.data["created_at"],
                "updated_at": self.data["updated_at"]
            }
        }

        # Parse JSON strings into dictionaries
        json_fields = {
            "issuer_details": json.loads(self.data["issuer_details"]),
            "instrument_details": json.loads(self.data["instrument_details"]),
            "coupon_details": json.loads(self.data["coupon_details"]),
            "redemption_details": json.loads(self.data["redemption_details"]),
            "credit_rating_details": json.loads(self.data["credit_rating_details"]),
            "listing_details": json.loads(self.data["listing_details"]),
            "key_contacts_details": json.loads(self.data["key_contacts_details"]),
            "key_documents_details": json.loads(self.data["key_documents_details"])
        }

        # Add non-null values from each JSON object
        for field_name, json_data in json_fields.items():
            parsed_data[field_name] = self._extract_non_null_values(json_data)

        return parsed_data

    def _extract_non_null_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively extract non-null values from nested dictionaries"""
        result = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    if isinstance(value, (dict, list)):
                        processed_value = self._extract_non_null_values(value)
                        if processed_value:  # Only add if there are non-null values
                            result[key] = processed_value
                    else:
                        result[key] = value
        elif isinstance(data, list):
            result = [
                self._extract_non_null_values(item) 
                for item in data 
                if item is not None
            ]
            result = [item for item in result if item]  # Remove empty dicts/lists
            
        return result

# Create parser and process the data
bond_data = pd.read_csv("bonds_details_202503011115.csv")

parser = BondDataParser(bond_data)
parsed_data = parser.parse_non_null_fields()

# Pretty print the results
print("Bond Data Analysis - Non-null Fields:")
print("\nBasic Information:")
for key, value in parsed_data["basic_info"].items():
    print(f"{key}: {value}")

for section, data in parsed_data.items():
    if section != "basic_info":
        print(f"\n{section.replace('_', ' ').title()}:")
        print(json.dumps(data, indent=2))