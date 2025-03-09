import os
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


class BondScreeneragent:
    def __init__(self, csv_path = "../../data/company_insights_202503011114.csv", model_name="llama3-70b-8192", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        load_dotenv()
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.model_name = model_name
        
        self.search_tool = TavilySearchResults(max_results=2)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
        self.retriever_tool = self._setup_retriever_tool()
        self.tools = [self.search_tool, self.retriever_tool]
        
        self.model = self._initialize_chat_model()
        self.agent_executor = self._initialize_agent_executor()
    
    def _setup_retriever_tool(self):
        loader = CSVLoader(file_path=self.csv_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        
        db = FAISS.from_documents(texts, self.embeddings)
        retriever = db.as_retriever()
        
        return create_retriever_tool(
            retriever,
            "Company Insider details tool",
            "Search for information about company-level financial analysis of bond-issuing firms."
        )
    
    def _initialize_chat_model(self):
        model = init_chat_model(self.model_name, model_provider="groq")
        model = model.bind_tools(self.tools)
        return model
    
    def _initialize_agent_executor(self):
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(self.model, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools)
    
    def query(self, input_query):
        response = self.agent_executor.invoke({"input": input_query})
        return response["output"]


if __name__ == "__main__":
    agent = BondScreeneragent()
    result = agent.query("Give an overview of the financial analysis of Urgo Capital Limited.")
    print(result)
