from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv
load_dotenv()

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=False,
)

# Using run() method to get the response as a string
def get_web_info(query):
    response = web_agent.run(query)
    message_content=response.content

    return message_content

# Example usage
if __name__ == "__main__":
    query = "What's happening in France?"
    result = get_web_info(query)
    
    # Print the result
    print(result)