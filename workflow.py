from orchestratoragent import BondQueryRouter
from bond_directory_agent import BondDirectoryAgent
from bond_screener_agent import BondScreeneragent
from Cashflow_agent import CompanyCashflow
from bond_finder_agent import BondFinderAgent
import pandas as pd
from websearch import WebAgent
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain.globals import set_llm_cache
import hashlib
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache


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


set_llm_cache(GPTCache(init_gptcache))

load_dotenv()

class BondWorkflowChain:
    def __init__(self):
        self.router = BondQueryRouter()
        self.directory = BondDirectoryAgent()
        self.screener = BondScreeneragent()
        self.cashflow = CompanyCashflow()
        self.finder = BondFinderAgent()
        self.llm = ChatGroq(model="llama3-70b-8192")
    
    def process_query(self, user_query):
        """
        Process a bond-related query through the appropriate chain of agents.
        
        Args:
            user_query (str): The user's bond-related query
            
        Returns:
            str: The final response from the appropriate agent
        """
        print(f"Processing query: {user_query}")
        
        routing_result = self.router.route_query(user_query)
        print(f"Router result: {routing_result}")
        
        if self._should_route_to_directory(routing_result) == 0:
            directory_result = self.directory.route_query(user_query)
            print(f"Directory result: {directory_result}")
            
            if self._should_route_to_finder(directory_result):
                return self._process_with_finder(user_query)
            else:
                return self._process_with_cashflow(user_query)
        else:
            return self._process_with_screener(user_query)
    
    def _should_route_to_directory(self, routing_result):
        """Determine if query should be routed to directory agent."""
        try:
            decision = int(str(routing_result).strip())
            return decision == 1
        except:
            return "1" in str(routing_result) or "directory" in str(routing_result).lower()
    
    def _should_route_to_finder(self, directory_result):
        """Determine if query should be routed to finder agent."""
        try:
            decision = int(str(directory_result).strip())
            return decision == 1
        except:
            return "1" in str(directory_result) or "finder" in str(directory_result).lower()
    
    def _process_with_screener(self, query):
        """Process the query with the bond screener agent."""
        result = self.screener.query(query)
        
        if isinstance(result, str):
            result_str = result
        else:
            result_str = str(result)
        
        print(f"Screener result: {result_str}...")  # Print first 100 chars
        return result_str
    
    def _process_with_cashflow(self, query):
        """Process the query with the cashflow agent."""
        result = self.cashflow.query(query)
        
        # Format result as string
        if isinstance(result, str):
            result_str = result
        else:
            result_str = str(result)
        
        print(f"Cashflow result: {result_str}...")  # Print first 100 chars
        return result_str
    
    def _process_with_finder(self, query):
        """Process the query with the bond finder agent."""
        try:
            result = self.finder.process_query(query)
            
            # If result is a dictionary with pandas DataFrame
            if isinstance(result, dict) and "results" in result and isinstance(result["results"], pd.DataFrame):
                df = result["results"]
                display_cols = ['isin', 'company_name', 'issue_size', 'maturity_date']
                display_cols = [col for col in display_cols if col in df.columns]
                
                result_str = f"Extracted keywords: {result.get('keywords', [])}\n\n"
                result_str += f"Found {result.get('count', 0)} matching bond(s):\n\n"
                
                if len(df) > 0:
                    result_str += df[display_cols].to_string(index=False)
                else:
                    result_str += "No matching bonds found."
            else:
                # Generic formatting
                result_str = str(result)
        except Exception as e:
            print(f"Error in finder_adapter: {e}")
            # Fallback to answer method or return error
            try:
                result = self.finder.answer(query)
                result_str = str(result)
            except:
                result_str = f"Error processing bond query: {str(e)}"
        
        print(f"Finder result: {result_str[:100]}...")  # Print first 100 chars
        return result_str


    def run_bond_workflow(self, user_query):
        """
        Process a user query through the bond workflow chain.
        
        Args:
            user_query (str): The user's bond-related query
            
        Returns:
            str: The final response
        """
        workflow = BondWorkflowChain()
        result = workflow.process_query(user_query)
        web_result = WebAgent().get_info(user_query)

        joiner_prompt = f"Structure the following two responses: \n\n1. {result} \n\n2. {web_result}"
        print("\nIntermediate Result:", result)
        messages = [
            ("system", "Generate a well-structured, coherent, and contextually accurate response based on the provided query. Ensure clarity, completeness, and logical flow while maintaining a professional and polished tone."),
            ("human", f"User query : {joiner_prompt}")
        ]
        answer = self.llm.invoke(messages)
        return answer.content


if __name__ == "__main__":
    user_query = "Give an overview of the financial analysis of Urgo Capital Limited"
    workflow = BondWorkflowChain()
    result = workflow.run_bond_workflow(user_query)
    print("\nFinal Response:")
    print(result)