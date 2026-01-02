# ============================================================================================
# from dotenv import load_dotenv
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# from langchain_community.tools.playwright.utils import create_sync_playwright_browser



# browser = create_sync_playwright_browser(headless=False)
# toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
# tools = toolkit.get_tools()


# tool_dict = {tool.name: tool for tool in tools}
# navigate_tool = tool_dict.get("navigate_browser")
# extract_text_tool = tool_dict.get("extract_text")
# navigate_tool.run({"url":"https://www.cnn.com"})
# text=extract_text_tool.run({})
# print(text)
# ============================================================================================

# I have to work on this. I have to fix the Async issue.




from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from pydantic import BaseModel
from typing import Annotated 
from langgraph.prebuilt import ToolNode, tools_condition
import gradio as gr


class State(BaseModel):        
    messages: Annotated[list, add_messages]

#==============================================================================
# Create the methods those will be used as tool
#==============================================================================

def send_information(text: str)-> str:
    """Send the notification via different channel as needed."""
    print(text)
    return "Information has been sent"




#==============================================================================
# Create the tool
#==============================================================================
def get_all_tools():
    # define all the tools
    tool_send = Tool(
        name="send_information",
        func=send_information,
        description="Useful tool to send notification via different channel as needed."
    )  
    
    browser = create_sync_playwright_browser(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
    tools = toolkit.get_tools()

    all_tools = tools + [tool_send]
    return all_tools

#==============================================================================
# Get llm and llm with tools
#==============================================================================
def get_llm():
    all_tools = get_all_tools()
    MODEL_NAME = os.getenv(key="MODEL_NAME")
    llm = ChatOpenAI(model=MODEL_NAME)
    llm_with_tools = llm.bind_tools(all_tools)
    return llm_with_tools



def chat_node(old_state: State) -> State:
    llm_with_tools = get_llm()
    response = llm_with_tools.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state


def setup_super_steps():
    
    all_tools = get_all_tools()
    # Start the Graphbuilder class with State Class
    graph_builder = StateGraph(State)

    # Define the nodes
    graph_builder.add_node("chatbot", chat_node)# create another node to call the tools
    graph_builder.add_node("tools", ToolNode(tools=all_tools))
    
    # Define the edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
    graph_builder.add_edge( "tools","chatbot")
    graph = graph_builder.compile()

    return graph


def chat(user_input: str, history):
    graph = setup_super_steps();
    message = {"role": "user", "content": user_input}
    messages = [message]
    state = State(messages=messages)
    result = graph.invoke(state)
    return result["messages"][-1].content


 
    


if __name__ == "__main__":

    load_dotenv()
    gr.ChatInterface(chat).launch()




