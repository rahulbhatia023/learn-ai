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
    description: str = Field(description="A detailed information of the product")
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


class ComparedProduct(BaseModel):
    recommended_product: RecommendedProduct
    price_score: float = Field(
        description="Is it reasonably priced considering its features and quality?"
    )
    value_score: float = Field(
        description="Does the product offer good value for its price?"
    )
    features_score: float = Field(
        description="Does it offer the features relevant to the user's query?"
    )
    specs_score: float = Field(
        description="Does the product meet the user's specifications?"
    )
    brand_score: float = Field(description="Is the brand reputable and trusted?")
    usp_score: float = Field(
        description="Does the product have any standout features or benefits?"
    )
    support_score: float = Field(
        description="Are there good after-sales services or warranties?"
    )
    relevance_score: float = Field(
        description="How well does the product match the user's criteria?"
    )
    overall_score: float = Field(
        description="The overall score calculated based on the individual attribute scores."
    )
    score_summary: str = Field(
        description="A brief summary of the product's overall score and why it was recommended."
    )


class ProductsComparison(BaseModel):
    products_comparison: List[ComparedProduct] = Field(
        description="The top 3 recommended products based on the comparison."
    )


class BestProduct(BaseModel):
    best_product: ComparedProduct = Field(
        description="The best product among the top 3 recommended products."
    )
    justification: str = Field(
        description="A brief explanation of why this product was chosen as the best option."
    )


class ShopWiseAgentState(TypedDict):
    query: str
    web_search_url: str
    web_search_content: str
    recommended_products: RecommendedProducts
    products_comparison: ProductsComparison
    best_product: BestProduct


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

        def compare_products(state: ShopWiseAgentState):
            prompt_template = """
                You are a professional assistant skilled in product analysis and comparison. Your task is to evaluate and compare the following products based on key attributes to recommend up to max 3 of the best options. 
    
                User Criteria:
                {user_query}
    
                Product Details:
                {recommended_products}
    
                Instructions:
                1. Assign a score between 1 and 10 for each attribute.
                2. Calculate an overall score for each product by averaging the attribute scores.
                3. Recommend up to 3 products with the highest overall scores. Provide a concise explanation for your recommendations, highlighting why they are the best fit for the user's needs.
            """

            response = llm.with_structured_output(ProductsComparison).invoke(
                PromptTemplate(
                    template=prompt_template,
                    input_variables=["user_query", "recommended_products"],
                ).format(
                    user_query=state["query"],
                    recommended_products=state["recommended_products"],
                )
            )

            return {"products_comparison": response.products_comparison[:3]}

        def best_product(state: ShopWiseAgentState):
            prompt_template = """
                You are an expert assistant specializing in product analysis and decision-making. Your task is to evaluate the top 3 recommended products and select the single best option based on their attributes and overall score. 

                Evaluation Details:
                User Criteria: 
                {user_query}
                
                Top 3 Products:
                {top3_products}

                Instructions:
                1. Carefully compare the top 3 products across the various available scores.                
                2. Select the best single option based on the overall score and the user's criteria. 
                3. Clearly explain why the selected product is the best option, referencing its strengths over the other two products.
            """

            response = llm.with_structured_output(BestProduct).invoke(
                PromptTemplate(
                    template=prompt_template,
                    input_variables=["user_query", "top3_products"],
                ).format(
                    user_query=state["query"],
                    top3_products=state["products_comparison"],
                )
            )

            return {"best_product": response}

        graph = StateGraph(ShopWiseAgentState)

        graph.add_node("Web Search", web_search)
        graph.add_node("Recommend Products", recommended_products)
        graph.add_node("Compare Products", compare_products)
        graph.add_node("Get Best Product", best_product)

        graph.add_edge(START, "Web Search")
        graph.add_edge("Web Search", "Recommend Products")
        graph.add_edge("Recommend Products", "Compare Products")
        graph.add_edge("Compare Products", "Get Best Product")
        graph.add_edge("Get Best Product", END)

        return graph.compile(
            checkpointer=MemorySaver(),
            interrupt_after=["Web Search", "Recommend Products", "Compare Products"],
        )
