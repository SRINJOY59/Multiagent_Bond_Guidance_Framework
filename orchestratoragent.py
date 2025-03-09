import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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


class BondQueryResponse(BaseModel):
    binary_score: int = Field(..., description="0 for Bond Directory Agent, 1 for Bond Screener Agent")

class BondQueryRouter:
    def __init__(self, model = "llama3-70b-8192"):
        """
        Initializes the BondQueryRouter with a language model.
        """
        self.llm = ChatGroq(model=model)
        self.structured_llm_router = self.llm.with_structured_output(BondQueryResponse)
        
        self.system_prompt = """
        You are an AI-powered Orchestrator Agent responsible for routing user queries to the correct specialized agent. 
        Your task is to determine if a query is related to:
        1. 'Bond Directory Agent' (0) – if the user is asking for ISIN-level bond details, credit ratings, maturity dates, or security types.
        2. 'Bond Screener Agent' (1) – if the user is asking for company-level financial analysis of bond-issuing firms or wants to filter bonds based on financial criteria.

        **Evaluation Criteria:**
        - **Return `0`** if the query involves:
          - ISIN-level bond details
          - Credit ratings and issuer details
          - Maturity dates and bond types

        - **Return `1`** if the query involves:
          - Financial analysis of bond-issuing firms
          - Company performance insights
          - Screening/filtering bonds based on financial metrics

        **Output Format:**
        - If the query is **about ISIN, credit ratings, or bond details**, output: `0`
        - If the query is **about financial analysis of bond issuers**, output: `1`

        **Examples:**
        🔹 **User Query:** "Give me the ISIN details for bond XYZ123."
        👉 **Output:** `0`

        🔹 **User Query:** "Show me financial analysis for bond-issuing companies."
        👉 **Output:** `1`

        🔹 **User Query:** "Find bonds with a high yield and good credit rating."
        👉 **Output:** `1`

        🔹 **User Query:** "What is the maturity date of bond ABC456?"
        👉 **Output:** `0`

        **Ensure that your response is only `0` or `1`, with no additional explanation.**
        """

        self.query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{query}"),
            ]
        )

        self.router = self.query_prompt | self.structured_llm_router

    def route_query(self, query):
        """
        Routes the bond-related query to either Bond Directory Agent (0) or Bond Screener Agent (1).
        """
        response = self.router.invoke({"query": query})
        return response.binary_score

if __name__ == "__main__":
    router = BondQueryRouter()

    # Sample Queries
    queries = [
        "Give me the ISIN details for bond XYZ123.",
        "Show me financial analysis for bond-issuing companies.",
        "Find bonds with a high yield and good credit rating.",
        "What is the maturity date of bond ABC456?"
    ]

    for query in queries:
        routing_result = router.route_query(query)
        print(f"Query: {query} -> Routing Decision: {routing_result}")  # Expected Output: 0 or 1
