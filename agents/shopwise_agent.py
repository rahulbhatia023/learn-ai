import random
from typing import TypedDict, Optional, List

import streamlit as st
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient


class RecommendedProduct(BaseModel):
    name: str = Field(description="The product name.")
    description: str = Field(
        description="A concise summary of the product's main features or purpose."
    )
    price: Optional[float] = Field(description="The product's price.")
    currency: str = Field(description="The currency in which the price is listed.")
    features: List[str] = Field(
        description="A list of key features or specifications of the product."
    )
    pros: List[str] = Field(
        description="A list of pros or advantages of the product, if available."
    )
    cons: List[str] = Field(
        description="A list of cons or disadvantages of the product, if available."
    )
    url: str = Field(description="Link to the product page.")
    image_url: Optional[List[str]] = Field(
        description="A link to an image of the product, if available."
    )


class RecommendedProducts(BaseModel):
    recommended_products: List[RecommendedProduct]


class ShopWiseAgentState(TypedDict):
    query: str
    web_search_url: str
    web_search_content: str
    recommended_products: RecommendedProducts


class ShopWiseAgent:
    agent_name = "ShopWise Agent"

    required_api_keys = {
        "TAVILY_API_KEY": "password",
        "OPENAI_API_KEY": "password",
        "FIRECRAWL_API_KEY": "password",
    }

    @classmethod
    def get_graph(cls):
        tavily_client = TavilyClient(api_key=st.session_state["TAVILY_API_KEY"])

        llm = ChatOpenAI(
            model_name="gpt-4o", openai_api_key=st.session_state["OPENAI_API_KEY"]
        )

        def web_search(state: ShopWiseAgentState):
            web_search_results = tavily_client.search(
                query=state["query"], max_results=3
            )["results"]

            web_search_result = web_search_results[
                random.randint(0, len(web_search_results) - 1)
            ]

            firecrawl_loader = FireCrawlLoader(
                api_key=st.session_state["FIRECRAWL_API_KEY"],
                url=web_search_result["url"],
                mode="scrape",
            )

            web_search_content = " ".join(
                [doc.page_content for doc in firecrawl_loader.lazy_load()]
            )

            return {
                "web_search_url": web_search_result["url"],
                "web_search_content": web_search_content,
            }

        def recommended_products(state: ShopWiseAgentState):
            web_search_content = state["web_search_content"]

            prompt_template = """
                You are a highly skilled assistant specialized in extracting structured information about shopping products from web search results. 
                Your task is to analyze the provided web search content and extract detailed product information while adhering to the user's specific filter criteria.
    
                For instance, if the user specifies preferences such as price range, brand, features, or other attributes, ensure that the extracted product details align with these requirements.
    
                Here is the user's query for context:
                {user_query}
    
                And here is the web search content for your analysis:
                {web_search_content}
    
                Your response should include only relevant product details that match the user's criteria. Ensure the extracted information is accurate, concise, and well-structured.
            """

            response = llm.with_structured_output(RecommendedProducts).invoke(
                PromptTemplate(
                    template=prompt_template,
                    input_variables=["user_query", "web_search_content"],
                ).format(
                    user_query=state["query"], web_search_content=web_search_content
                )
            )

            return {"recommended_products": response.recommended_products}

        graph = StateGraph(ShopWiseAgentState)

        graph.add_node("Web Search", web_search)
        graph.add_node("Recommended Products", recommended_products)

        graph.add_edge(START, "Web Search")
        graph.add_edge("Web Search", "Recommended Products")
        graph.add_edge("Recommended Products", END)

        return graph.compile(checkpointer=MemorySaver(), interrupt_after=["Web Search"])
