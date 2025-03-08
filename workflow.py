from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END
from orchestratoragent import BondQueryRouter
from bond_directory_agent import BondDirectoryAgent
from bond_screener_agent import BondScreeneragent
from Cashflow_agent import CashflowAgent
from bond_finder_agent import BondFinderAgent
from langchain_core.messages import HumanMessage, AIMessage


graph_builder = StateGraph(MessagesState)


graph_builder.add_node("router", BondQueryRouter.route_query)
graph_builder.add_node("directory", BondDirectoryAgent.route_query)
graph_builder.add_node("screener", BondScreeneragent.query)
graph_builder.add_node("cashflow", CashflowAgent.query)
graph_builder.add_node("finder", BondFinderAgent.query)


graph_builder.set_entry_point("router")


def router_condition(state):

    messages = state["messages"]
    last_message = messages[-1]
    

    content = last_message.content
    
    try:
        decision = int(content.strip())
        if decision == 1:
            return "directory"
        else:
            return "screener"
    except:
        if "1" in content or "directory" in content.lower():
            return "directory"
        else:
            return "screener"


def directory_condition(state):

    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content
    
    try:
        decision = int(content.strip())
        if decision == 1:
            return "finder"
        else:
            return "cashflow"
    except:
        if "1" in content or "finder" in content.lower():
            return "finder"
        else:
            return "cashflow"


graph_builder.add_conditional_edges(
    "router",
    router_condition,
    {
        "directory": "directory",
        "screener": "screener"
    }
)

graph_builder.add_conditional_edges(
    "directory",
    directory_condition,
    {
        "finder": "finder",
        "cashflow": "cashflow"
    }
)

graph_builder.add_edge("finder", END)
graph_builder.add_edge("cashflow", END)
graph_builder.add_edge("screener", END)


graph = graph_builder.compile()


def run_bond_workflow(user_query):
    messages = [HumanMessage(content=user_query)]
    
    initial_state = {"messages": messages}
    
    
    result = graph.invoke(initial_state)
    
    
    final_message = result["messages"][-1]
    return final_message.content

if __name__ == "__main__":
    user_query = "Tell me about corporate bonds with high yield"
    result = run_bond_workflow(user_query)
    print("\nFinal Response:")
    print(result)
    
