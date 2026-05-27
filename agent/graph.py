from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tools import analyze_dataframe, execute_python_code, read_csv_file

TOOLS = [read_csv_file, analyze_dataframe, execute_python_code]

_SYSTEM_PROMPT = """\
You are an expert data analyst and Python programmer.
The user has uploaded a CSV file at: {file_path}

Use your tools to help the user analyze their data:
- read_csv_file    — preview the data structure and column types
- analyze_dataframe — comprehensive stats: nulls, duplicates, value counts
- execute_python_code — write and run pandas/matplotlib code for deeper analysis

When calling execute_python_code:
  • Always pass file_path="{file_path}"
  • 'df' is pre-loaded; use it directly
  • Use print() for any values you want displayed
  • Create matplotlib/seaborn charts normally (never call plt.show())

Workflow for analysis requests:
  1. Read/profile the data first if you haven't yet
  2. Write Python code to investigate and visualize
  3. Explain findings clearly in plain English
  4. If asked to clean/fix data, provide complete, runnable Python code

Be precise and analytical. Show code whenever generating transformations or fixes.\
"""


def build_graph() -> StateGraph:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        system = SystemMessage(
            content=_SYSTEM_PROMPT.format(file_path=state.get("file_path", ""))
        )
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())
