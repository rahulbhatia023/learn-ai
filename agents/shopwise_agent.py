import json
import time
from typing import List, Optional, TypedDict

import streamlit as st
from googleapiclient.discovery import build
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import Field, BaseModel
from tavily import TavilyClient


class SpecsComparison(BaseModel):
    processor: str = Field(
        ..., description="Processor type and model, e.g., 'Snapdragon 888'"
    )
    battery: str = Field(..., description="Battery capacity and type, e.g., '4500mAh'")
    camera: str = Field(..., description="Camera specs, e.g., '108MP primary'")
    display: str = Field(
        ...,
        description="Display type, size, refresh rate, e.g., '6.5 inch OLED, 120Hz'",
    )
    storage: str = Field(
        ..., description="Storage options and expandability, e.g., '128GB, expandable'"
    )


class RatingsComparison(BaseModel):
    overall_rating: float = Field(..., description="Overall rating out of 5, e.g., 4.5")
    performance: float = Field(
        ..., description="Rating for performance out of 5, e.g., 4.7"
    )
    battery_life: float = Field(
        ..., description="Rating for battery life out of 5, e.g., 4.3"
    )
    camera_quality: float = Field(
        ..., description="Rating for camera quality out of 5, e.g., 4.6"
    )
    display_quality: float = Field(
        ..., description="Rating for display quality out of 5, e.g., 4.8"
    )


class Comparison(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    specs_comparison: SpecsComparison
    ratings_comparison: RatingsComparison
    reviews_summary: str = Field(
        ..., description="Summary of key points from user reviews about this product"
    )


class BestProduct(BaseModel):
    product_name: str = Field(..., description="Name of the best product")
    justification: str = Field(
        ..., description="Explanation of why this product is the best choice"
    )


class ProductComparison(BaseModel):
    comparisons: List[Comparison]
    best_product: BestProduct


class Highlights(BaseModel):
    Camera: Optional[str] = None
    Performance: Optional[str] = None
    Display: Optional[str] = None
    Fast_Charging: Optional[str] = None


class SmartphoneReview(BaseModel):
    title: str = Field(..., description="The title of the smartphone review")
    url: Optional[str] = Field(None, description="The URL of the smartphone review")
    content: Optional[str] = Field(
        None, description="The main content of the smartphone review"
    )
    pros: Optional[List[str]] = Field(None, description="The pros of the smartphone")
    cons: Optional[List[str]] = Field(None, description="The cons of the smartphone")
    highlights: Optional[dict] = Field(
        None, description="The highlights of the smartphone"
    )
    score: Optional[float] = Field(None, description="The score of the smartphone")


class ListOfSmartphoneReviews(BaseModel):
    reviews: List[SmartphoneReview] = Field(
        ..., description="List of individual smartphone reviews"
    )


class ShopWiseAgentState(TypedDict):
    query: str
    products: list[dict]
    product_schema: list[SmartphoneReview]
    blogs_content: Optional[List[dict]]
    best_product: dict
    comparison: list
    youtube_link: str


class ShopWiseAgent:
    agent_name = "ShopWise Agent"

    required_api_keys = {
        "TAVILY_API_KEY": "password",
        "YOUTUBE_API_KEY": "password",
        "OPENAI_API_KEY": "password",
    }

    @classmethod
    def get_graph(cls):
        llm = ChatOpenAI(
            model_name="gpt-4o", openai_api_key=st.session_state["OPENAI_API_KEY"]
        )

        tavily_client = TavilyClient(api_key=st.session_state["TAVILY_API_KEY"])

        youtube_client = build(
            "youtube", "v3", developerKey=st.session_state["YOUTUBE_API_KEY"]
        )

        def load_blog_content(page_url):
            return " ".join(
                [
                    doc.page_content
                    for doc in WebBaseLoader(
                        web_paths=[page_url],
                        bs_get_text_kwargs={"separator": " ", "strip": True},
                    ).load()
                ]
            )

        def tavily_search_node(state: ShopWiseAgentState):
            response = tavily_client.search(query=state["query"], max_results=1)
            blogs_content = []
            for blog in response["results"]:
                blog_url = blog.get("url", "")
                if blog_url:
                    content = load_blog_content(blog_url)
                    if content:
                        blogs_content.append(
                            {
                                "title": blog.get("title", ""),
                                "url": blog_url,
                                "content": content,
                                "score": blog.get("score", ""),
                            }
                        )
            return {"blogs_content": blogs_content}

        def schema_mapping_node(state: ShopWiseAgentState):
            max_retries = 2
            wait_time = 60
            blogs_content = state["blogs_content"]
            prompt_template = """
                You are a professional assistant tasked with extracting structured information from a blogs.

                ### Instructions:
                
                1. **Product Details**: For each product mentioned in the blog post, populate the `products` array with structured data for each item, including:
                   - `title`: The product name.
                   - `url`: Link to the blog post or relevant page.
                   - `content`: A concise summary of the product's main features or purpose.
                   - `pros`: A list of positive aspects or advantages of the product.if available other wise extract blog content.
                   - `cons`: A list of negative aspects or disadvantages.if available other wise extract blog content.
                   - `highlights`: A dictionary containing notable features or specifications.if available other wise extract blog content.
                   - `score`: A numerical rating score if available; otherwise, use `0.0`.
                
                ### Blogs Contents: {blogs_content}
                
                After extracting all information, just return the response in the JSON structure given below. Do not add any extracted information. The JSON should be in a valid structure with no extra characters inside, like Python’s \n.
            """

            parser = JsonOutputParser(pydantic_object=ListOfSmartphoneReviews)

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["blogs_content"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            for attempt in range(1, max_retries + 1):
                chain = prompt | llm | parser
                response = chain.invoke({"blogs_content": blogs_content})

                if response.get("products") and len(response["products"]) > 1:
                    return {"product_schema": response["products"]}

                if attempt < max_retries:
                    time.sleep(wait_time)

            return {"product_schema": []}

        def product_comparison_node(state: ShopWiseAgentState):
            if "product_schema" in state and state["product_schema"]:
                prompt_template = """
                                1. **List of Products for Comparison (`comparisons`):**
                                   - Each product should include:
                                     - **Product Name**: The name of the product (e.g., "Smartphone A").
                                     - **Specs Comparison**:
                                       - **Processor**: Type and model of the processor (e.g., "Snapdragon 888").
                                       - **Battery**: Battery capacity and type (e.g., "4500mAh").
                                       - **Camera**: Camera specifications (e.g., "108MP primary").
                                       - **Display**: Display type, size, and refresh rate (e.g., "6.5 inch OLED, 120Hz").
                                       - **Storage**: Storage options and whether it is expandable (e.g., "128GB, expandable").
                                     - **Ratings Comparison**:
                                       - **Overall Rating**: Overall rating out of 5 (e.g., 4.5).
                                       - **Performance**: Rating for performance out of 5 (e.g., 4.7).
                                       - **Battery Life**: Rating for battery life out of 5 (e.g., 4.3).
                                       - **Camera Quality**: Rating for camera quality out of 5 (e.g., 4.6).
                                       - **Display Quality**: Rating for display quality out of 5 (e.g., 4.8).
                                     - **Reviews Summary**: Summary of key points from user reviews that highlight the strengths and weaknesses of this product.
                        
                                2. **Best Product Selection (`best_product`):**
                                   - **Product Name**: Select the best product among the compared items.
                                   - **Justification**: Provide a brief explanation of why this product is considered the best choice. This should be based on factors such as balanced performance, high user ratings, advanced specifications, or unique features.
                                
                                ---
                        
                                ### Example Output:
                        
                                ```json
                                {{
                                  "comparisons": [
                                    {{
                                      "product_name": "Smartphone A",
                                      "specs_comparison": {{
                                        "processor": "Snapdragon 888",
                                        "battery": "4500mAh",
                                        "camera": "108MP primary",
                                        "display": "6.5 inch OLED, 120Hz",
                                        "storage": "128GB, expandable"
                                      }},
                                      "ratings_comparison": {{
                                        "overall_rating": 4.5,
                                        "performance": 4.7,
                                        "battery_life": 4.3,
                                        "camera_quality": 4.6,
                                        "display_quality": 4.8
                                      }},
                                      "reviews_summary": "Highly rated for display quality and camera performance, with a strong processor. Battery life is good but may drain faster with heavy use."
                                    }},
                                    {{
                                      "product_name": "Smartphone B",
                                      "specs_comparison": {{
                                        "processor": "Apple A15 Bionic",
                                        "battery": "4000mAh",
                                        "camera": "12MP Dual",
                                        "display": "6.1 inch Super Retina XDR, 60Hz",
                                        "storage": "256GB, non-expandable"
                                      }},
                                      "ratings_comparison": {{
                                        "overall_rating": 4.6,
                                        "performance": 4.8,
                                        "battery_life": 4.1,
                                        "camera_quality": 4.5,
                                        "display_quality": 4.7
                                      }},
                                      "reviews_summary": "Smooth user experience with excellent performance and display. The battery is slightly smaller but generally sufficient for moderate use."
                                    }}
                                  ],
                                  "best_product": {{
                                    "product_name": "Smartphone A",
                                    "justification": "Chosen for its high-quality display, strong camera, and balanced performance that meets most user needs."
                                  }}
                                }}
                        
                                ```
                                Here is the product data to analyze:
                                {product_data}
                            """

                parser = JsonOutputParser(pydantic_object=ProductComparison)

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["product_data"],
                    partial_variables={
                        "format_instructions": parser.get_format_instructions()
                    },
                )

                chain = prompt | llm | parser

                response = chain.invoke(
                    {"product_data": json.dumps(state["product_schema"])}
                )

                return {
                    "comparison": response["comparisons"],
                    "best_product": response["best_product"],
                }
            else:
                return state

        def youtube_review_node(state: ShopWiseAgentState):
            best_product_name = state.get("best_product", {}).get("product_name")

            if not best_product_name:
                return {"youtube_link": None}

            search_response = (
                youtube_client.search()
                .list(
                    q=f"{best_product_name} review",
                    part="snippet",
                    type="video",
                    maxResults=1,
                )
                .execute()
            )

            video_items = search_response.get("items", [])

            if not video_items:
                return {"youtube_link": None}

            video_id = video_items[0]["id"]["videoId"]
            youtube_link = f"https://www.youtube.com/watch?v={video_id}"

            return {"youtube_link": youtube_link}

        graph = StateGraph(ShopWiseAgentState)

        graph.add_node("Web Search", tavily_search_node)
        graph.add_node("Schema Mapping", schema_mapping_node)
        graph.add_node("Products Comparison", product_comparison_node)
        graph.add_node("Youtube Review", youtube_review_node)

        graph.add_edge(START, "Web Search")
        graph.add_edge("Web Search", "Schema Mapping")
        graph.add_edge("Schema Mapping", "Products Comparison")
        graph.add_edge("Products Comparison", "Youtube Review")
        graph.add_edge("Youtube Review", END)

        return graph.compile(
            checkpointer=MemorySaver(), interrupt_after=["Schema Mapping"]
        )
