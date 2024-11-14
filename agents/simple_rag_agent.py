from typing import Sequence

import streamlit as st
from langchain.prompts import Prompt
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools.simple_rag_tools import DocumentsRetrieverTool


class SimpleRAGAgent:
    agent_name = "Simple RAG Agent"

    system_prompt = """
    You are a helpful assistant. Answer the user's questions based on the tools provided.
    """

    nodes_to_display = ["agent", "generate"]

    @classmethod
    def get_tools(cls) -> Sequence[BaseTool]:
        return [
            DocumentsRetrieverTool(
                pdf_file=st.session_state["uploaded_file"][cls.agent_name],
                openai_api_key=st.session_state["OPENAI_API_KEY"],
            )
        ]

    @classmethod
    def get_graph(cls) -> CompiledStateGraph:
        if (
            "uploaded_file" in st.session_state
            and cls.agent_name in st.session_state["uploaded_file"]
            and st.session_state["uploaded_file"][cls.agent_name]
        ):
            tools = cls.get_tools()
        else:
            tools = []

        llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=st.session_state["OPENAI_API_KEY"],
            temperature=0,
        )

        llm_with_tools = llm.bind_tools(tools)

        def agent(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        def generate(state):
            messages = state["messages"]
            question = messages[0].content
            docs = messages[-1].content

            prompt = Prompt.from_template(
                """
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                Question: {question} 
                
                Context: {context} 
                
                Answer:
                """
            )

            rag_chain = prompt | llm_with_tools

            response = rag_chain.invoke({"context": docs, "question": question})

            return {"messages": [response]}

        graph = StateGraph(MessagesState)

        graph.add_node("agent", agent)
        graph.add_node("retrieve", ToolNode(tools))
        graph.add_node("generate", generate)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", tools_condition, {"tools": "retrieve", END: END}
        )
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()
