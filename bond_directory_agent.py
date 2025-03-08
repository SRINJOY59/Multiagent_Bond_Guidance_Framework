from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

class BondQueryResponse(BaseModel):
    binary_score: int = Field(..., description="1 for Bond Finder Agent, 0 for Cash Flow & Maturity Agent")

class BondDirectoryAgent:
    def __init__(self, model):
        """
        Initializes the BondQueryRouter with a language model and API key.
        """
        self.llm = ChatGroq(model=model)
        self.structured_llm_router = self.llm.with_structured_output(BondQueryResponse)
        
        self.system_prompt = """
        You are an evaluator determining whether a bond-related query should be routed to:
        - The **Bond Finder Agent (1)** for queries about bond comparison, best yields, and investment options.
        - The **Cash Flow & Maturity Agent (0)** for queries related to cash flows, maturity schedules, and payment timelines.

        **Evaluation Criteria:**
        - **Return `1`** if the query involves:
          - Bond comparison
          - Finding the best yield or highest return
          - Selecting bonds from multiple platforms
          - Investment recommendations based on yields
          - Platform comparisons for bond purchases

        - **Return `0`** if the query involves:
          - Bond cash flows
          - Maturity schedules and payment timelines
          - Next coupon or interest payment date
          - Detailed breakdown of cash flows over time

        **Output Format:**
        - If the query is **about bond comparison and yields**, output: `1`
        - If the query is **about cash flows, maturity schedules, or payment timelines**, output: `0`

        **Examples:**
        🔹 **User Query:** "Which platform offers the best corporate bond yields?"
        👉 **Output:** `1`

        🔹 **User Query:** "When is the next payment for bond ISIN XYZ?"
        👉 **Output:** `0`

        🔹 **User Query:** "Compare bond yields across multiple platforms."
        👉 **Output:** `1`

        🔹 **User Query:** "Show me the cash flow structure for bond ABC."
        👉 **Output:** `0`

        🔹 **User Query:** "List government bonds with the highest returns available today."
        👉 **Output:** `1`

        **Ensure that your response is only `1` or `0`, with no additional explanation.**
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
        Routes the bond-related query to either Bond Finder Agent (1) or Cash Flow & Maturity Agent (0).
        """
        response = self.router.invoke({"query": query})
        return response.binary_score



if __name__ == "__main__":
    model = "llama3-70b-8192"


    router = BondDirectoryAgent(model)

    # Sample Query
    query = "Show me details for ISIN INE 123456789."
    routing_result = router.route_query(query)

    print(f"Routing Decision: {routing_result}")  # Expected Output: 1
