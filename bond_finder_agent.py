from langchain_groq import ChatGroq
import os
import json
import logging
import re
import time
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BondFinderAgent:
    """
    Process bond queries by extracting keywords using LLMs and searching a bond database.
    """
    
    def __init__(self, 
                 llm_model_name: str = "llama3-70b-8192", 
                 api_key: Optional[str] = None,
                 csv_path: str = "bonds_details_202503011115.csv"):
        """
        Initialize the Bond Query Processor.
        
        Args:
            llm_model_name: Name of the LLM model to use for keyword extraction
            api_key: Groq API key (will use environment variable if None)
            csv_path: Path to the CSV file containing bond details
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
        
        # Store CSV path
        self.csv_path = csv_path
        self.bond_data = None
        
        # Load bond data from CSV
        self._load_bond_data()
        
        logger.info(f"BondQueryProcessor initialized with model: {llm_model_name}")
    
    def _load_bond_data(self):
        """
        Load bond data from the CSV file.
        """
        try:
            logger.info(f"Loading bond data from {self.csv_path}")
            self.bond_data = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded {len(self.bond_data)} bond records")
            
            # Convert column names to lowercase for case-insensitive matching
            self.bond_data.columns = [col.lower() for col in self.bond_data.columns]
            
            # Check if necessary columns exist
            required_columns = ['isin', 'issuer', 'company_name']
            available_columns = [col for col in required_columns if col in self.bond_data.columns]
            
            if not available_columns:
                logger.warning(f"CSV doesn't contain any of the expected columns: {required_columns}")
            else:
                logger.info(f"Available key columns: {available_columns}")
                
        except FileNotFoundError:
            logger.error(f"Bond data file not found: {self.csv_path}")
            self.bond_data = None
        except Exception as e:
            logger.error(f"Error loading bond data: {str(e)}")
            self.bond_data = None
    
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
    
    def search_bonds(self, keywords: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Search for bonds in the CSV data based on extracted keywords.
        
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
        
        # If we have an ISIN code, filter by it first (highest priority)
        if keywords.get("isin_code"):
            isin = keywords["isin_code"]
            
            # Check if we have an ISIN column
            if 'isin' in self.bond_data.columns:
                search_results = search_results[search_results['isin'].str.contains(isin, case=False, na=False)]
                search_criteria.append(f"ISIN: {isin}")
            else:
                logger.warning("No ISIN column found in CSV, skipping ISIN search")
        
        # If no results found with ISIN or no ISIN provided, try company name
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
        
        # If we still don't have results, return all data with a note
        if search_results.empty and not search_criteria:
            logger.info("No specific search criteria applied, returning all bonds")
            search_results = self.bond_data
            search_criteria.append("No specific criteria - showing all bonds")
        
        logger.info(f"Search returned {len(search_results)} results using criteria: {search_criteria}")
        return search_results, search_criteria
    
    def format_bond_results(self, results: pd.DataFrame, max_results: int = 10) -> str:
        """
        Format bond search results into a readable string.
        
        Args:
            results: DataFrame containing bond search results
            max_results: Maximum number of results to include in the output
            
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
        
        # Get key columns to display
        # Priority order of columns we'd like to show
        priority_columns = [
            "isin","company_name","issue_size","allotment_date","maturity_date", "coupon_details",
        ]
        
        # Filter to columns that actually exist in our data
        display_columns = [col for col in priority_columns if col in results.columns]
        
        # If none of our priority columns exist, use all columns
        if not display_columns:
            display_columns = results.columns
        
        # Format each bond as a readable entry
        count = 0
        for idx, bond in results[display_columns].iterrows():
            output += f"Bond {idx+1}:\n"
            for col in display_columns:
                if count >= 3:
                    break
                if not pd.isna(bond[col]):
                    # Format column name for readability
                    col_display = col.replace('_', ' ').title()
                    output += f"  {col_display}: {bond[col]}\n"
                count += 1
            output += "\n"
        
        return output
    
    def process_query(self, user_query: str, rag_system=None) -> str:
        """
        Process a bond query using extracted keywords and search bond_details.csv.
        
        Args:
            user_query: User's query about bonds
            rag_system: Optional RAG system for advanced queries (if None, uses CSV search only)
            
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
        
        # If RAG system is provided and we want more detailed answers
        rag_response = ""
        if rag_system and not search_results.empty:
            # Enhance the query with extracted keywords
            enhanced_query = user_query
            
            # If ISIN code was found, prioritize it in search
            if extracted_keywords.get("isin_code"):
                isin_code = extracted_keywords["isin_code"]
                enhanced_query = f"ISIN:{isin_code} {user_query}"
            
            # If company name was found and no ISIN, prioritize it
            elif extracted_keywords.get("company_name"):
                company = extracted_keywords["company_name"]
                enhanced_query = f"company:{company} {user_query}"
            
            logger.info(f"Sending enhanced query to RAG system: {enhanced_query}")
            rag_response = rag_system.answer_query(enhanced_query)
        
        # Format the final response
        final_response = f"""
Extracted Keywords:
- ISIN/INE Code: {extracted_keywords.get('isin_code', 'None detected')}
- Company: {extracted_keywords.get('company_name', 'None detected')}

Search Criteria Applied: {', '.join(search_criteria)}

Bond Search Results:
{formatted_results}
"""

        # Add RAG response if available
        if rag_response:
            final_response += f"\nAdditional Information:\n{rag_response}"
        
        logger.info(f"Query processing completed in {time.time() - start_time:.2f} seconds")
        messages = [
            ("system", ""
            "You have to extract the coupon details properly from the give user query and return the extracted coupon details in the response and give the complete details in the response."), 
            ("user", final_response)
        ]
        response = self.llm.invoke(messages)
        return response.content


def main():
    """
    Main function to demonstrate the BondQueryProcessor.
    """


    user_query = "Describe details for INE002A08534?"
    print(f"\nProcessing query: {user_query}")
    
    try:
        processor = BondFinderAgent()
        
        result = processor.process_query(user_query)
        print(result)
        

    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    main()