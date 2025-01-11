from typing import Dict, Any, List, TypedDict

import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from common.sqlite import get_schema, execute_query


class DataQueryState(TypedDict):
    question: str
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    query_columns: List[str]
    sql_valid: bool
    sql_issues: str
    results: List[Any]
    answer: str


class DataQueryAgent:
    agent_name = "Data Query Agent"

    system_prompt = """
            You are a data analyst that can help summarize SQL tables and parse user questions about a database.
        """

    required_api_keys = {"OPENAI_API_KEY": "password"}

    @classmethod
    def get_graph(cls):
        llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=st.session_state["OPENAI_API_KEY"],
            temperature=0,
        )

        def parse_question(state: DataQueryState):
            """Parse user question and identify relevant tables and columns."""
            question = state["question"]

            schema = get_schema(
                sqlite_file=st.session_state["uploaded_file"][cls.agent_name]
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                                You are a data analyst that can help summarize SQL tables and parse user questions about a database. 
                                Given the question and database schema, identify the relevant tables and columns. 
                                If the question is not relevant to the database or if there is not enough information to answer the question, set is_relevant to false.
        
                                Your response should be in the following JSON format:
                                {{
                                    "is_relevant": boolean,
                                    "relevant_tables": [
                                        {{
                                            "table_name": string,
                                            "columns": [string],
                                            "noun_columns": [string]
                                        }}
                                    ]
                                }}
        
                                The "noun_columns" field should contain only the columns that are relevant to the question and contain nouns or names, 
                                for example, the column "Artist name" contains nouns relevant to the question "What are the top selling artists?", but the column "Artist ID" is not relevant because it does not contain a noun. 
                                Do not include columns that contain numbers.""",
                    ),
                    (
                        "human",
                        """
                        Database schema:
                        {schema}
                        
                        User question:
                        {question}
                        
                        Identify relevant tables and columns""",
                    ),
                ]
            )

            response = llm.invoke(
                prompt.format_messages(schema=schema, question=question)
            ).content

            parsed_response = JsonOutputParser().parse(response)

            return {"parsed_question": parsed_response}

        def get_unique_nouns(state: DataQueryState):
            """Find unique nouns in relevant tables and columns."""
            parsed_question = state["parsed_question"]

            if not parsed_question["is_relevant"]:
                return {"unique_nouns": []}

            unique_nouns = set()
            for table_info in parsed_question["relevant_tables"]:
                table_name = table_info["table_name"]
                noun_columns = table_info["noun_columns"]

                if noun_columns:
                    column_names = ", ".join(f"`{col}`" for col in noun_columns)
                    query = f"SELECT DISTINCT {column_names} FROM `{table_name}`"
                    results = execute_query(
                        sqlite_file=st.session_state["uploaded_file"][cls.agent_name],
                        query=query,
                    )
                    for row in results:
                        unique_nouns.update(str(value) for value in row if value)

            return {"unique_nouns": list(unique_nouns)}

        def generate_sql(state: DataQueryState) -> dict:
            """Generate SQL query based on parsed question and unique nouns."""
            question = state["question"]
            parsed_question = state["parsed_question"]
            unique_nouns = state["unique_nouns"]

            if not parsed_question["is_relevant"]:
                return {"sql_query": "NOT_RELEVANT", "is_relevant": False}

            schema = get_schema(
                sqlite_file=st.session_state["uploaded_file"][cls.agent_name]
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are an AI assistant that generates SQL queries based on user questions, database schema, and unique nouns found in the relevant tables. Generate a valid SQL query to answer the user's question.
        
                        If there is not enough information to write a SQL query, respond with "NOT_ENOUGH_INFO".
        
                        Here are some examples:
        
                        1. What is the top selling product?
                        Answer: SELECT product_name, SUM(quantity) as total_quantity FROM sales GROUP BY product_name ORDER BY total_quantity DESC LIMIT 1
        
                        2. What is the total revenue for each product?
                        Answer: SELECT \`product name\`, SUM(quantity * price) as total_revenue FROM sales GROUP BY \`product name\`  ORDER BY total_revenue DESC
        
                        3. What is the market share of each product?
                        Answer: SELECT \`product name\`, SUM(quantity) * 100.0 / (SELECT SUM(quantity) FROM sa  les) as market_share FROM sales GROUP BY \`product name\`  ORDER BY market_share DESC
        
                        Just give the query string. Do not format it. Make sure to use the correct spellings of nouns as provided in the unique nouns list. All the table and column names should be enclosed in backticks.
                        """,
                    ),
                    (
                        "human",
                        """
                            Database schema:
                            {schema}
            
                            User question:
                            {question}
            
                            Relevant tables and columns:
                            {parsed_question}
            
                            Unique nouns in relevant tables:
                            {unique_nouns}
            
                            Respond in JSON format with the following structure. Only respond with the JSON:
                            {{
                                "sql_query": string,
                                "query_columns": [string]
                            }}
                            
                            The "query_columns" field should contain the columns used in the SQL query. 
                            Consider alias columns as well. For example, if you use "SUM(quantity) as total_quantity", then "total_quantity" should be included in the "query_columns" list.
                            
                            If the response is NOT_ENOUGH_INFO, then output should be: 
                            {{
                                "sql_query": "NOT_RELEVANT"
                                "query_columns": None
                            }}
                        """,
                    ),
                ]
            )

            response = llm.invoke(
                prompt.format_messages(
                    schema=schema,
                    question=question,
                    parsed_question=parsed_question,
                    unique_nouns=unique_nouns,
                )
            ).content

            result = JsonOutputParser().parse(response)

            return {
                "sql_query": result["sql_query"],
                "query_columns": result["query_columns"],
            }

        def validate_and_fix_sql(state: DataQueryState) -> dict:
            """Validate and fix the generated SQL query."""
            sql_query = state["sql_query"]

            if sql_query == "NOT_RELEVANT":
                return {"sql_query": "NOT_RELEVANT", "sql_valid": False}

            schema = get_schema(
                sqlite_file=st.session_state["uploaded_file"][cls.agent_name]
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                            You are an AI assistant that validates and fixes SQL queries. Your task is to:
                            1. Check if the SQL query is valid.
                            2. Ensure all table and column names are correctly spelled and exist in the schema. All the table and column names should be enclosed in backticks.
                            3. If there are any issues, fix them and provide the corrected SQL query.
                            4. If no issues are found, return the original query.
        
                            Respond in JSON format with the following structure. Only respond with the JSON:
                            {{
                                "valid": boolean,
                                "issues": string or null,
                                "corrected_query": string
                            }}
                            
                            For example:
                            1. {{
                                "valid": true,
                                "issues": null,
                                "corrected_query": "None"
                            }}
        
                            2. {{
                                "valid": false,
                                "issues": "Column USERS does not exist",
                                "corrected_query": "SELECT * FROM \`users\` WHERE age > 25"
                            }}
        
                            3. {{
                                "valid": false,
                                "issues": "Column names and table names should be enclosed in backticks if they contain spaces or special characters",
                                "corrected_query": "SELECT * FROM \`gross income\` WHERE \`age\` > 25"
                            }}
                            """,
                    ),
                    (
                        "human",
                        """
                            Database schema:
                            {schema}
        
                            Generated SQL query:
                            {sql_query}
                        """,
                    ),
                ]
            )

            response = llm.invoke(
                prompt.format_messages(schema=schema, sql_query=sql_query)
            ).content

            result = JsonOutputParser().parse(response)

            if result["valid"] and result["issues"] is None:
                return {"sql_query": sql_query, "sql_valid": True}
            else:
                return {
                    "sql_query": result["corrected_query"],
                    "sql_valid": result["valid"],
                    "sql_issues": result["issues"],
                }

        def execute_sql(state: DataQueryState) -> dict:
            """Execute SQL query and return results."""
            query = state["sql_query"]

            if query == "NOT_RELEVANT":
                return {"results": "NOT_RELEVANT"}

            results = execute_query(
                sqlite_file=st.session_state["uploaded_file"][cls.agent_name],
                query=query,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                            You are an AI assistant that analyse the results and returns the final answer summary for the given query.
                            Give the final conclusion based on the results in 1-2 lines. 
                        """,
                    ),
                    (
                        "human",
                        """
                            Query:
                            {query}
                            
                            Results:
                            {results}
                        """,
                    ),
                ]
            )

            answer = llm.invoke(
                prompt.format_messages(query=query, results=results)
            ).content

            return {"results": results, "answer": answer}

        graph = StateGraph(state_schema=DataQueryState)

        graph.add_node("parse_question", parse_question)
        graph.add_node("get_unique_nouns", get_unique_nouns)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("validate_and_fix_sql", validate_and_fix_sql)
        graph.add_node("execute_sql", execute_sql)

        graph.add_edge(START, "parse_question")
        graph.add_edge("parse_question", "get_unique_nouns")
        graph.add_edge("get_unique_nouns", "generate_sql")
        graph.add_edge("generate_sql", "validate_and_fix_sql")
        graph.add_edge("validate_and_fix_sql", "execute_sql")
        graph.add_edge("execute_sql", END)

        return graph.compile(checkpointer=MemorySaver())
