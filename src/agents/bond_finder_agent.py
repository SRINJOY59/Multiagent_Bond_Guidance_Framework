from langchain_groq import ChatGroq
import os
import json
import logging
import re
import time
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz  # For fuzzy matching

# Load environment variables
load_dotenv()

from langchain.globals import set_llm_cache
import hashlib
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache


# Load environment variables
load_dotenv()
def get_hashed_name(name):
    """Generate a hashed name for the LLM model to use in cache storage."""
    return hashlib.sha256(name.encode()).hexdigest()
def init_gptcache(cache_obj: Cache, llm: str):
    """Initialize the GPTCache system."""
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


# Initialize GPT Cache
set_llm_cache(GPTCache(init_gptcache))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BondFinderAgent:
    """
    Process bond queries by extracting keywords using LLMs and searching a bond database.
    Includes fuzzy search capability and handles complex nested JSON structures.
    """
    
    def __init__(self, 
                 llm_model_name: str = "llama3-70b-8192", 
                 api_key: Optional[str] = None,
                 csv_path: str = "../../data/bonds_details_202503011115.csv",
                 fuzzy_threshold: int = 80):  # Threshold for fuzzy matching (0-100)
        """
        Initialize the Bond Query Processor.
        
        Args:
            llm_model_name: Name of the LLM model to use for keyword extraction
            api_key: Groq API key (will use environment variable if None)
            csv_path: Path to the CSV file containing bond details
            fuzzy_threshold: Minimum score for fuzzy matching (0-100, higher is stricter)
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set and no API key provided.")
        
        # Initialize the LLM client
        self.llm = ChatGroq(model_name=llm_model_name, api_key=api_key)
        
        # Regex patterns for ISIN/INE codes
        self.isin_pattern = r'\b(?:ISIN:?\s*)?([A-Z]{2}[A-Z0-9]{9}[0-9])\b'
        self.ine_pattern = r'\b(?:INE[A-Z0-9]{7}[0-9])\b'
        
        # Store CSV path and fuzzy threshold
        self.csv_path = csv_path
        self.bond_data = None
        self.bond_data_expanded = None  # Will store the expanded dataframe with parsed JSON
        self.fuzzy_threshold = fuzzy_threshold
        
        # Load bond data from CSV
        self._load_bond_data()
        
        logger.info(f"BondFinderAgent initialized with model: {llm_model_name}")
    
    def _parse_json_column(self, df: pd.DataFrame, column_name: str, prefix: str = '') -> pd.DataFrame:
        """
        Parse a JSON column in the DataFrame and expand it into separate columns.
        
        Args:
            df: DataFrame containing the column to parse
            column_name: Name of the column containing JSON data
            prefix: Prefix to add to the expanded column names
            
        Returns:
            pd.DataFrame: DataFrame with expanded columns
        """
        if column_name not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        try:
            # Check if the column contains JSON strings or already parsed dictionaries
            if df[column_name].dtype == 'object':
                sample_value = df[column_name].iloc[0] if not df.empty else None
                
                if isinstance(sample_value, str):
                    # Parse JSON strings
                    parsed_series = df[column_name].apply(lambda x: json.loads(x) if pd.notna(x) and x else {})
                elif isinstance(sample_value, dict):
                    # Already a dictionary
                    parsed_series = df[column_name]
                else:
                    # Not parseable
                    return df
                
                # Convert the Series of dictionaries to a DataFrame
                json_df = pd.json_normalize(parsed_series)
                
                if json_df.empty:
                    return df
                
                # Add prefix to column names
                if prefix:
                    json_df.columns = [f"{prefix}_{col}" for col in json_df.columns]
                
                # Drop the original column
                result_df = result_df.drop(columns=[column_name])
                
                # Return the combined DataFrame
                return pd.concat([result_df, json_df], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing JSON in column {column_name}: {str(e)}")
            return df
    
    def _expand_nested_json(self, df: pd.DataFrame, max_depth: int = 3) -> pd.DataFrame:
        """
        Recursively expand nested JSON structures in the DataFrame up to a maximum depth.
        
        Args:
            df: DataFrame to expand
            max_depth: Maximum depth for recursion
            
        Returns:
            pd.DataFrame: Expanded DataFrame
        """
        if max_depth <= 0:
            return df
        
        result_df = df.copy()
        
        json_columns = []
        for col in result_df.columns:
            if result_df[col].dtype == 'object':
                sample_value = result_df[col].iloc[0] if not result_df.empty else None
                if isinstance(sample_value, str) and sample_value.startswith('{') and sample_value.endswith('}'):
                    json_columns.append(col)
                elif isinstance(sample_value, dict):
                    json_columns.append(col)
        
        if not json_columns:
            return result_df
            
        # Parse each JSON column
        for col in json_columns:
            result_df = self._parse_json_column(result_df, col)
            
        # Recursively process for nested structures
        return self._expand_nested_json(result_df, max_depth - 1)
    
    def _load_bond_data(self):
        """
        Load bond data from the CSV file and parse nested JSON structures.
        """
        try:
            logger.info(f"Loading bond data from {self.csv_path}")
            self.bond_data = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded {len(self.bond_data)} bond records")
            
            # Convert column names to lowercase for case-insensitive matching
            self.bond_data.columns = [col.lower() for col in self.bond_data.columns]
            
            # Process JSON columns to expand nested structures
            json_columns = [
                'issuer_details', 'instrument_details', 'coupon_details', 
                'redemption_details', 'credit_rating_details', 'listing_details', 
                'key_contacts_details'
            ]
            
            # Make a copy for expanded data
            self.bond_data_expanded = self.bond_data.copy()
            
            # Parse each JSON column
            for col in json_columns:
                if col in self.bond_data.columns:
                    logger.info(f"Parsing JSON in column: {col}")
                    try:
                        # Try to parse as JSON if it's a string
                        if self.bond_data_expanded[col].dtype == 'object':
                            self.bond_data_expanded[col] = self.bond_data_expanded[col].apply(
                                lambda x: json.loads(x) if isinstance(x, str) and pd.notna(x) else x
                            )
                    except Exception as e:
                        logger.warning(f"Error parsing JSON in column {col}: {str(e)}")
            
            # Extract coupon details as a separate flattened structure
            if 'coupon_details' in self.bond_data.columns:
                logger.info("Extracting coupon details")
                coupon_info = []
                
                for idx, row in self.bond_data.iterrows():
                    coupon_details = row.get('coupon_details')
                    isin = row.get('isin')
                    coupon_info_dict = {'isin': isin}
                    
                    try:
                        if isinstance(coupon_details, str):
                            coupon_details = json.loads(coupon_details)
                        
                        if isinstance(coupon_details, dict):
                            # Extract coupon rate and frequency
                            coupons_vo = coupon_details.get('coupensVo', {})
                            if coupons_vo:
                                coupon_info_dict['coupon_rate'] = coupons_vo.get('rate')
                                coupon_info_dict['coupon_frequency'] = coupons_vo.get('frequency')
                                coupon_info_dict['coupon_type'] = coupons_vo.get('type')
                                
                                # Extract step up/down details
                                step_status = coupens_vo.get('stepStatus', {})
                                if step_status:
                                    coupon_info_dict['step_up'] = step_status.get('stepUp')
                                    coupon_info_dict['step_down'] = step_status.get('stepDown')
                    except Exception as e:
                        logger.warning(f"Error processing coupon details for row {idx}: {str(e)}")
                    
                    coupon_info.append(coupon_info_dict)
                
                # Create a DataFrame with extracted coupon info
                self.coupon_info_df = pd.DataFrame(coupon_info)
                logger.info(f"Created coupon info DataFrame with {len(self.coupon_info_df)} records")
            
            # Check if necessary columns exist
            required_columns = ['isin', 'issuer', 'company_name']
            available_columns = [col for col in required_columns if col in self.bond_data.columns]
            
            if not available_columns:
                logger.warning(f"CSV doesn't contain any of the expected columns: {required_columns}")
            else:
                logger.info(f"Available key columns: {available_columns}")
                
            # Create a dictionary of ISIN values for faster fuzzy matching
            if 'isin' in self.bond_data.columns:
                self.isin_values = self.bond_data['isin'].dropna().unique().tolist()
                logger.info(f"Loaded {len(self.isin_values)} unique ISIN values for fuzzy matching")
            else:
                self.isin_values = []
                
            # Create a dictionary of company names for faster fuzzy matching
            company_cols = [col for col in self.bond_data.columns if 'company' in col or 'issuer' in col]
            self.company_values = []
            for col in company_cols:
                values = self.bond_data[col].dropna().unique().tolist()
                self.company_values.extend(values)
            self.company_values = list(set(self.company_values))
            logger.info(f"Loaded {len(self.company_values)} unique company/issuer values for fuzzy matching")
                
        except FileNotFoundError:
            logger.error(f"Bond data file not found: {self.csv_path}")
            self.bond_data = None
            self.bond_data_expanded = None
        except Exception as e:
            logger.error(f"Error loading bond data: {str(e)}")
            self.bond_data = None
            self.bond_data_expanded = None
    
    def extract_keywords(self, query: str) -> Dict[str, Any]:
        """
        Extract ISIN codes, INE codes, or company names from a bond query using an LLM.
        
        Args:
            query: User's bond-related query
            
        Returns:
            dict: Extracted keywords (isin_code, company_name)
        """
        start_time = time.time()
        logger.info(f"Extracting keywords from query: {query}")
        
        # Extract using regex first (as backup)
        isin_matches = re.findall(self.isin_pattern, query, re.IGNORECASE)
        ine_matches = re.findall(self.ine_pattern, query, re.IGNORECASE)
        regex_codes = list(set(isin_matches + ine_matches))
        
        if regex_codes:
            logger.info(f"Found ISIN/INE codes via regex: {regex_codes}")
        
        # Create a simple prompt for the LLM
        system_prompt = "You are a financial data extraction assistant. Extract ISIN codes and company names from queries about bonds."
        
        user_prompt = f"""
        From the following query about bonds, extract:
        1. Any ISIN codes (like INE002A08534)
        2. Any company names
        
        Query: {query}
        
        Return ONLY a JSON object with these fields:
        {{
            "isin_code": "extracted ISIN or INE code (or null if none)",
            "company_name": "extracted company name (or null if none)"
        }}
        """
        
        try:
            # Send to LLM
            messages = [
                ("system", system_prompt),
                ("user", user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Try to extract JSON from response
            try:
                # Extract JSON pattern from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group(0))
                else:
                    extracted_data = json.loads(content)
                    
                # Merge with regex results if needed
                if regex_codes and (not extracted_data.get("isin_code") or extracted_data["isin_code"] == "null"):
                    extracted_data["isin_code"] = regex_codes[0]
                    
                logger.info(f"Extracted data: {extracted_data}")
                logger.info(f"Keyword extraction completed in {time.time() - start_time:.2f} seconds")
                return extracted_data
                
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from LLM response, using regex results")
                result = {
                    "isin_code": regex_codes[0] if regex_codes else None,
                    "company_name": None
                }
                logger.info(f"Keyword extraction completed in {time.time() - start_time:.2f} seconds")
                return result
                
        except Exception as e:
            logger.error(f"Error extracting keywords with LLM: {str(e)}")
            result = {
                "isin_code": regex_codes[0] if regex_codes else None,
                "company_name": None,
                "error": str(e)
            }
            logger.info(f"Keyword extraction completed in {time.time() - start_time:.2f} seconds")
            return result
    
    def fuzzy_match_isin(self, isin_code: str) -> str:
        """
        Find the closest matching ISIN code using fuzzy matching.
        
        Args:
            isin_code: The ISIN code to match
            
        Returns:
            str: The closest matching ISIN code from the database, or original if no match
        """
        if not isin_code or not self.isin_values:
            return isin_code
        
        # Use process.extractOne to find the best match
        match = process.extractOne(isin_code, self.isin_values, scorer=fuzz.ratio)
        
        if match and match[1] >= self.fuzzy_threshold:
            logger.info(f"Fuzzy matched ISIN {isin_code} to {match[0]} with score {match[1]}")
            return match[0]
        
        logger.info(f"No good fuzzy match found for ISIN {isin_code} (best score: {match[1] if match else 'N/A'})")
        return isin_code
    
    def extract_coupon_details(self, bond_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract and format the coupon details from the bond data.
        
        Args:
            bond_data: DataFrame containing bond data with JSON coupon_details column
            
        Returns:
            dict: Structured coupon details
        """
        if bond_data.empty:
            return {}
        
        try:
            # Get the first row (assuming we're looking at a specific bond)
            row = bond_data.iloc[0]
            coupon_details = row.get('coupon_details')
            
            # Default empty structure
            result = {
                "coupon_rate": None,
                "coupon_type": None,
                "frequency": None,
                "day_count": None,
                "step_up": None,
                "step_down": None
            }
            
            # Parse the coupon details
            if isinstance(coupon_details, str):
                try:
                    coupon_details = json.loads(coupon_details)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse coupon_details JSON: {coupon_details}")
                    return result
            
            if isinstance(coupon_details, dict):
                coupens_vo = coupon_details.get('coupensVo', {})
                if coupens_vo:
                    result["coupon_rate"] = coupens_vo.get('rate')
                    result["coupon_type"] = coupens_vo.get('type')
                    result["frequency"] = coupens_vo.get('frequency')
                    result["day_count"] = coupens_vo.get('daycount')
                    
                    step_status = coupens_vo.get('stepStatus', {})
                    if step_status:
                        result["step_up"] = step_status.get('stepUp')
                        result["step_down"] = step_status.get('stepDown')
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting coupon details: {str(e)}")
            return {}
    
    def search_bonds(self, keywords: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Search for bonds in the CSV data based on extracted keywords, using fuzzy matching.
        
        Args:
            keywords: Dictionary containing extracted keywords
            
        Returns:
            tuple: (DataFrame with search results, list of search criteria used)
        """
        if self.bond_data is None or self.bond_data.empty:
            logger.error("Bond data not available for searching")
            return pd.DataFrame(), ["No bond data available"]
        
        search_results = self.bond_data.copy()
        search_criteria = []
        
        # Apply fuzzy matching to keywords
        if keywords.get("isin_code"):
            original_isin = keywords["isin_code"]
            keywords["isin_code"] = self.fuzzy_match_isin(keywords["isin_code"])
            
            if original_isin != keywords["isin_code"]:
                search_criteria.append(f"Fuzzy matched ISIN: {original_isin} → {keywords['isin_code']}")
        
        # If we have an ISIN code, filter by it (highest priority)
        if keywords.get("isin_code"):
            isin = keywords["isin_code"]
            
            # Check if we have an ISIN column
            if 'isin' in self.bond_data.columns:
                search_results = search_results[search_results['isin'].str.contains(isin, case=False, na=False)]
                search_criteria.append(f"ISIN: {isin}")
            else:
                logger.warning("No ISIN column found in CSV, skipping ISIN search")
        
        # If no results found or no ISIN provided, try company name
        if (search_results.empty or not keywords.get("isin_code")) and keywords.get("company_name"):
            company = keywords["company_name"]
            
            # Look for company name in various possible columns
            company_columns = [col for col in self.bond_data.columns if 'company' in col or 'issuer' in col]
            
            if company_columns:
                company_filter = pd.DataFrame(False, index=self.bond_data.index, columns=[0])
                
                for col in company_columns:
                    company_filter = company_filter | self.bond_data[col].str.contains(company, case=False, na=False)
                
                search_results = self.bond_data[company_filter[0]]
                search_criteria.append(f"Company: {company}")
            else:
                logger.warning("No company/issuer columns found in CSV, skipping company search")
        
        logger.info(f"Search returned {len(search_results)} results using criteria: {search_criteria}")
        return search_results, search_criteria
    
    def format_bond_results(self, results: pd.DataFrame, max_results: int = 10, 
                           include_coupon_details: bool = True) -> str:
        """
        Format bond search results into a readable string.
        
        Args:
            results: DataFrame containing bond search results
            max_results: Maximum number of results to include in the output
            include_coupon_details: Whether to parse and include detailed coupon information
            
        Returns:
            str: Formatted results
        """
        if results.empty:
            return "No matching bonds found."
        
        # Limit results to prevent overwhelming responses
        if len(results) > max_results:
            output = f"Found {len(results)} matching bonds. Showing top {max_results}:\n\n"
            results = results.head(max_results)
        else:
            output = f"Found {len(results)} matching bonds:\n\n"
        
        # Priority order of columns we'd like to show
        priority_columns = [
            "isin", "company_name", "issue_size", "allotment_date", 
            "maturity_date", "coupon_details"
        ]
        
        # Filter to columns that actually exist in our data
        display_columns = [col for col in priority_columns if col in results.columns]
        
        # If none of our priority columns exist, use all columns
        if not display_columns:
            display_columns = results.columns
        
        # Format each bond
        for idx, bond in results.iterrows():
            output += f"Bond {idx+1}:\n"
            output += "-" * 50 + "\n"
            
            # Display basic bond information
            for col in display_columns:
                if col != "coupon_details" and not pd.isna(bond[col]):
                    col_display = col.replace('_', ' ').title()
                    output += f"  {col_display}: {bond[col]}\n"
            
            # If requested, extract and display coupon details
            if include_coupon_details and 'coupon_details' in results.columns:
                try:
                    # Create a single-row DataFrame with this bond
                    single_bond_df = pd.DataFrame([bond])
                    coupon_info = self.extract_coupon_details(single_bond_df)
                    
                    if coupon_info:
                        output += "  Coupon Information:\n"
                        if coupon_info.get("coupon_rate"):
                            output += f"    - Rate: {coupon_info['coupon_rate']}\n"
                        if coupon_info.get("coupon_type"):
                            output += f"    - Type: {coupon_info['coupon_type']}\n"
                        if coupon_info.get("frequency"):
                            output += f"    - Frequency: {coupon_info['frequency']}\n"
                except Exception as e:
                    logger.error(f"Error formatting coupon details for bond {idx}: {str(e)}")
            
            output += "\n"
        
        return output
    
    def process_query(self, user_query: str, rag_system=None) -> str:
        """
        Process a bond query using extracted keywords and search bond_details.csv.
        
        Args:
            user_query: User's query about bonds
            rag_system: Optional RAG system for advanced queries
            
        Returns:
            str: Response to the query with bond information
        """
        start_time = time.time()
        logger.info(f"Processing query: {user_query}")
        
        # Extract keywords from query
        extracted_keywords = self.extract_keywords(user_query)
        logger.info(f"Extracted keywords: {extracted_keywords}")
        
        # Search for bonds based on keywords
        search_results, search_criteria = self.search_bonds(extracted_keywords)
        
        # Format the search results
        formatted_results = self.format_bond_results(search_results)
        
        # Build the final response for the LLM to enhance
        final_response = f"""
Extracted Keywords:
- ISIN/INE Code: {extracted_keywords.get('isin_code', 'None detected')}
- Company: {extracted_keywords.get('company_name', 'None detected')}

Search Criteria Applied: {', '.join(search_criteria)}

Bond Search Results:
{formatted_results}
"""
        
        # For INE002A08534 (or any matched ISIN), add special handling to extract all coupon details
        if extracted_keywords.get('isin_code') and not search_results.empty:
            try:
                # Extract coupon details from JSON
                coupon_details = None
                if 'coupon_details' in search_results.columns:
                    first_row = search_results.iloc[0]
                    coupon_details = first_row.get('coupon_details')
                    
                    if isinstance(coupon_details, str):
                        try:
                            coupon_details = json.loads(coupon_details)
                            
                            # Add detailed coupon information if available
                            if coupon_details:
                                final_response += "\nDetailed Coupon Information:\n"
                                final_response += json.dumps(coupon_details, indent=2)
                        except:
                            pass
            except Exception as e:
                logger.error(f"Error extracting detailed coupon information: {str(e)}")
        
        # Use LLM to format and enhance the response
        logger.info(f"Sending results to LLM for formatting")
        
        # Create the prompt for the LLM to enhance the response
        system_prompt = """
        You are a financial data expert specializing in bond information. 
        You have to extract the coupon details properly from the bond search results and 
        return the extracted coupon details in the response. Give the complete details in the response.
        Focus on parsing any JSON structures in the coupon_details field and present them in a readable format.
        """
        
        messages = [
            ("system", system_prompt), 
            ("user", final_response)
        ]
        
        try:
            response = self.llm.invoke(messages)
            enhanced_response = response.content
            
            logger.info(f"Query processing completed in {time.time() - start_time:.2f} seconds")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response with LLM: {str(e)}")
            return final_response
        
if __name__ == "__main__":
    agent = BondFinderAgent()
    result = agent.process_query("Show me details for ISIN INE 123456789.")
    print(result)