import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal
import pandas as pd

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults


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


class CashFlowValidationError(Exception):
    """Custom exception for cash flow validation errors"""

    pass


class ISINNotFoundError(Exception):
    """Custom exception for ISIN not found errors"""

    pass


class CompanyCashflow:
    def __init__(
        self, 
        csv_path: str = "../../data/cashflows_202503011113.csv", 
        model_name: str = "llama3-70b-8192",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the CompanyCashflow class with enhanced validation and data processing.

        Args:
            csv_path (str): Path to the CSV file containing cash flow data
            model_name (str): Name of the LLM model to use
            embedding_model (str): Name of the embedding model to use
        """
        self._validate_file(csv_path)
        load_dotenv()

        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.data = self._load_and_validate_data()

        # Initialize tools and models
        self.search_tool = TavilySearchResults(max_results=2)
        self.embeddings = self._initialize_embeddings()
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.retriever_tool = self._setup_retriever_tool()
        self.tools = [self.search_tool, self.retriever_tool]

        # Initialize model and agent
        self.model = self._initialize_chat_model()
        self.agent_executor = self._initialize_agent_executor()

        # Cache for frequently accessed data
        self._cache = {}

    def _validate_file(self, csv_path: str) -> None:
        """Validate the CSV file existence and format"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        if not csv_path.endswith(".csv"):
            raise ValueError("File must be a CSV file")

    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate the CSV data"""
        try:
            df = pd.read_csv(self.csv_path)
            required_columns = [
                "id",
                "isin",
                "cash_flow_date",
                "cash_flow_amount",
                "record_date",
                "principal_amount",
                "interest_amount",
                "tds_amount",
                "remaining_principal",
                "state",
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise CashFlowValidationError(
                    f"Missing required columns: {missing_cols}"
                )

            # Convert date columns
            date_columns = ["cash_flow_date", "record_date", "created_at", "updated_at"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Convert numeric columns
            numeric_columns = [
                "cash_flow_amount",
                "principal_amount",
                "interest_amount",
                "tds_amount",
                "remaining_principal",
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            raise CashFlowValidationError(f"Error loading CSV data: {str(e)}")

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model with error handling"""
        try:
            return HuggingFaceEmbeddings(model_name=self.embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

    def _setup_retriever_tool(self):
        """Set up the retriever tool with enhanced document processing"""
        try:
            loader = CSVLoader(file_path=self.csv_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(docs)

            db = FAISS.from_documents(texts, self.embeddings)
            retriever = db.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            return create_retriever_tool(
                retriever,
                "Cashflow details tool",
                "Search for information about Cashflow analysis of bond-issuing firms.",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup retriever tool: {str(e)}")

    def _initialize_chat_model(self):
        """Initialize the chat model with error handling"""
        try:
            model = init_chat_model(self.model_name, model_provider="groq")
            return model.bind_tools(self.tools)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chat model: {str(e)}")

    def _initialize_agent_executor(self):
        """Initialize the agent executor with error handling"""
        try:
            prompt = hub.pull("hwchase17/openai-functions-agent")
            agent = create_tool_calling_agent(self.model, self.tools, prompt)
            return AgentExecutor(agent=agent, tools=self.tools)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize agent executor: {str(e)}")

    def _get_isin_details(self, isin: str) -> Dict[str, Any]:
        """Get detailed information for a specific ISIN"""
        if isin in self._cache:
            return self._cache[isin]

        isin_data = self.data[self.data["isin"] == isin]
        if isin_data.empty:
            raise ISINNotFoundError(f"No data found for ISIN: {isin}")

        details = {
            "isin": isin,
            "total_principal": isin_data["principal_amount"].sum(),
            "total_interest": isin_data["interest_amount"].sum(),
            "next_payment_date": isin_data[
                isin_data["cash_flow_date"] > datetime.now()
            ]["cash_flow_date"].min(),
            "maturity_date": isin_data["cash_flow_date"].max(),
            "payment_schedule": isin_data[
                [
                    "cash_flow_date",
                    "cash_flow_amount",
                    "principal_amount",
                    "interest_amount",
                ]
            ]
            .sort_values("cash_flow_date")
            .to_dict("records"),
        }

        self._cache[isin] = details
        return details

    def query(self, input_query: str) -> str:
        """
        Process user queries using RAG for enhanced context
        """
        try:
            # Get relevant context using the retriever
            context = self.retriever_tool.run(input_query)

            if "ISIN" in input_query.upper():
                isin = input_query.upper().split("ISIN")[-1].strip()
                try:
                    details = self._get_isin_details(isin)
                    # Enhance response with retrieved context
                    base_response = self._format_isin_response(details)
                    return f"{base_response}\n\nAdditional Context:\n{context}"
                except ISINNotFoundError:
                    return f"No data found for ISIN: {isin}"

            # Use the agent with retrieved context
            enhanced_query = (
                f"Based on this context: {context}\n\nAnswer this query: {input_query}"
            )
            response = self.agent_executor.invoke({"input": enhanced_query})
            return response["output"]

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def _format_isin_response(self, details: Dict[str, Any]) -> str:
        """Format ISIN details into a readable response"""
        next_payment = (
            details["next_payment_date"].strftime("%Y-%m-%d")
            if details["next_payment_date"]
            else "No upcoming payments"
        )
        maturity = details["maturity_date"].strftime("%Y-%m-%d")

        response = f"""
ISIN Details: {details['isin']}
-------------------
Total Principal: {details['total_principal']:,.2f}
Total Interest: {details['total_interest']:,.2f}
Next Payment Date: {next_payment}
Maturity Date: {maturity}

Payment Schedule:
"""

        for payment in details["payment_schedule"]:
            response += f"\n{payment['cash_flow_date'].strftime('%Y-%m-%d')}: "
            response += f"Total: {payment['cash_flow_amount']:,.2f} "
            response += f"(Principal: {payment['principal_amount']:,.2f}, "
            response += f"Interest: {payment['interest_amount']:,.2f})"

        return response

    def get_upcoming_maturities(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get bonds maturing between specified dates"""
        maturities = []
        for isin in self.data["isin"].unique():
            isin_data = self.data[self.data["isin"] == isin]
            maturity_date = isin_data["cash_flow_date"].max()

            if start_date <= maturity_date <= end_date:
                maturities.append(
                    {
                        "isin": isin,
                        "maturity_date": maturity_date,
                        "final_payment": isin_data.iloc[-1]["cash_flow_amount"],
                        "total_principal": isin_data["principal_amount"].sum(),
                    }
                )

        return sorted(maturities, key=lambda x: x["maturity_date"])


# if __name__ == "__main__":
#     try:
#         agent = CompanyCashflow("cashflows.csv")

#         # Example queries
#         queries = [
#             "Show me details for ISIN INE0OOQ07320",
#             "Which bonds are maturing in 2025?",
#             "Show me the cash flow schedule for ISIN INE0B7Y07076"
#         ]

#         for query in queries:
#             print(f"\nQuery: {query}")
#             result = agent.query(query)
#             print(result)

#     except Exception as e:
#         print(f"Error running the program: {str(e)}")


if __name__ == "__main__":
    try:
        
        agent = CompanyCashflow()
        
        while True:
            user_query = input("\nEnter your query (or 'exit' to quit): ").strip()

            if user_query.lower() == "exit":
                break

            if user_query:
                result = agent.query(user_query)
                print(f"\n{result}")

    except Exception as e:
        print(f"Error: {str(e)}")
