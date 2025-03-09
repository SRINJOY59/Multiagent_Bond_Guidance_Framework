import os
import json
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

class BondFinderAgent:
    def __init__(
        self, 
        csv_path="../../data/bonds_details_cleaned.csv",
        model_name="llama3-70b-8192",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):
        load_dotenv()
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.model_name = model_name
        
        # Preprocess the CSV data
        self.processed_data = self._preprocess_csv()
        
        # Initialize tools and models
        self.search_tool = TavilySearchResults(max_results=2)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
        self.retriever_tool = self._setup_retriever_tool()
        self.tools = [self.search_tool, self.retriever_tool]
        
        self.model = self._initialize_chat_model()
        self.agent_executor = self._initialize_agent_executor()

    def _preprocess_csv(self):
        """Preprocess the CSV file"""
        # Read CSV with the new column structure
        df = pd.read_csv(self.csv_path)
        
        # Ensure all expected columns are present
        expected_columns = [
            'ISIN',
            'Issuer',
            'Face Value',
            'Face Value After PO',
            'Trade FV',
            'Units',
            'Trade Date',
            'YTM',
            'Coupon'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
                
        return df

    def _setup_retriever_tool(self):
        """Set up the retriever tool with processed data"""
        docs = []
        for _, row in self.processed_data.iterrows():
            doc_content = f"""
            ISIN: {row['ISIN']}
            Issuer: {row['Issuer']}
            Face Value: {row['Face Value']}
            Face Value After PO: {row['Face Value After PO']}
            Trade FV: {row['Trade FV']}
            Units: {row['Units']}
            Trade Date: {row['Trade Date']}
            YTM: {row['YTM']}
            Coupon: {row['Coupon']}
            """
            docs.append(doc_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents(docs)
        
        db = FAISS.from_documents(texts, self.embeddings)
        retriever = db.as_retriever()
        
        return create_retriever_tool(
            retriever,
            "Bond Trading Details",
            "Search for detailed information about bonds and their trading details."
        )

    def _initialize_chat_model(self):
        """Initialize the chat model with tools"""
        model = init_chat_model(self.model_name, model_provider="groq")
        model = model.bind_tools(self.tools)
        return model

    def _initialize_agent_executor(self):
        """Initialize the agent executor"""
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(self.model, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools)

    def get_bond_details_by_isin(self, isin):
        """Get detailed bond information for a specific ISIN"""
        bond_data = self.processed_data[self.processed_data['ISIN'] == isin].iloc[0]
        return {
            'ISIN': bond_data['ISIN'],
            'Issuer': bond_data['Issuer'],
            'Face Value': bond_data['Face Value'],
            'Face Value After PO': bond_data['Face Value After PO'],
            'Trade FV': bond_data['Trade FV'],
            'Units': bond_data['Units'],
            'Trade Date': bond_data['Trade Date'],
            'YTM': bond_data['YTM'],
            'Coupon': bond_data['Coupon']
        }

    def query(self, input_query):
        """Process queries about bond trading information"""
        response = self.agent_executor.invoke({"input": input_query})
        return response["output"]

def export_bond_data_to_csv(agent, isin, output_path):
    """Export bond data for a specific ISIN to a CSV file"""
    bond_data = agent.get_bond_details_by_isin(isin)
    
    # Convert to DataFrame and export to CSV
    bond_df = pd.DataFrame([bond_data])
    bond_df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    # Initialize the agent with default CSV path
    agent = BondFinderAgent()
    
    # Example query
    result = agent.query("What is the YTM and Face Value for ISIN INE002A08534?")
    print(result)