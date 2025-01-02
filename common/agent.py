from typing import Sequence

import streamlit as st
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class BaseAgent:
    name: str = None
    system_prompt: str = "You are a useful AI assistant."
    interrupt_before: list[str] = []
    update_as_node: str = None
    nodes_to_display = []
    tools: Sequence[BaseTool] = []

    model = "gpt-4o"

    @classmethod
    def get_tools(cls) -> Sequence[BaseTool]:
        return []

    @classmethod
    def update_graph_state(cls, human_message):
        return {}

    @classmethod
    def get_graph(cls):
        tools = cls.get_tools()

        llm = ChatOpenAI(
            model_name=cls.model,
            openai_api_key=st.session_state["OPENAI_API_KEY"],
            temperature=0,
        )

        if tools:
            llm = llm.bind_tools(tools=tools)

        def call_llm(state: MessagesState):
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)

        graph.add_node("agent", call_llm)
        graph.add_node("tools", ToolNode(tools=tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            source="agent",
            path=tools_condition,
            path_map={"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile(
            interrupt_before=cls.interrupt_before, checkpointer=MemorySaver()
        )
