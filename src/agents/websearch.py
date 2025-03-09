from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv

class WebAgent:
    def __init__(self):
        load_dotenv()
        self.agent = Agent(
            name="Web Agent",
            model=Groq(id="llama3-70b-8192"),
            tools=[DuckDuckGo()],
            instructions=["Always include sources"],
            show_tool_calls=True,
            markdown=False,
        )

    def get_info(self, query: str) -> str:
        response = self.agent.run(query)
        return response.content

if __name__ == "__main__":
    web_agent = WebAgent()
    query = "What's happening in France?"
    result = web_agent.get_info(query)
    print(result)
