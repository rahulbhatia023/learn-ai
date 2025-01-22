from typing import List

import streamlit as st

from agents.shopwise_agent import (
    ShopWiseAgent,
    RecommendedProduct,
    ComparedProduct,
    BestProduct,
)
from common.langgraph import add_langgraph_workflow_visualization
from common.page import get_api_key, keys_missing
from common.theme import set_page_config, container_title


def display_recommended_products(
    recommended_products: list[RecommendedProduct], number_of_products_in_a_row: int
):
    for i in range(0, len(recommended_products), number_of_products_in_a_row):
        products_in_a_row = recommended_products[i : i + number_of_products_in_a_row]
        cols = st.columns(number_of_products_in_a_row)
        for j in range(0, len(products_in_a_row)):
            with cols[j]:
                recommended_product = products_in_a_row[j]
                with st.container(border=True):
                    container_title(recommended_product.name)
                    if recommended_product.image_url:
                        st.image(
                            recommended_product.image_url, use_container_width=True
                        )
                    st.html(
                        f"<a href='{recommended_product.url}' style='font-family:Poppins; text-align:center'><p style='font-family:Poppins; text-align:center'>View Product</p></a>"
                    )
                    st.html(
                        f"<p style='font-family:Poppins; text-align:justify'>{recommended_product.description}</p>"
                    )
                    st.info(
                        f"PRICE: {recommended_product.price} {recommended_product.currency}"
                    )
                    with st.expander("Features", expanded=True):
                        for feature in recommended_product.features:
                            st.html(f"<li style='font-family:Poppins'>{feature}</li>")
                    with st.expander("Pros"):
                        for pro in recommended_product.pros:
                            st.html(f"<li style='font-family:Poppins'>{pro}</li>")
                    with st.expander("Cons"):
                        for con in recommended_product.cons:
                            st.html(f"<li style='font-family:Poppins'>{con}</li>")


def display_top3_products(
    top3_products: List[ComparedProduct], number_of_products_in_a_row: int
):
    for i in range(0, len(top3_products), number_of_products_in_a_row):
        products_in_a_row = top3_products[i : i + number_of_products_in_a_row]
        cols = st.columns(number_of_products_in_a_row)
        for j in range(0, len(products_in_a_row)):
            with cols[j]:
                product = products_in_a_row[j]
                with st.container(border=True):
                    container_title(product.recommended_product.name)
                    if product.recommended_product.image_url:
                        st.image(
                            product.recommended_product.image_url,
                            use_container_width=True,
                        )
                    st.html(
                        f"<a href='{product.recommended_product.url}' style='font-family:Poppins; text-align:center'><p style='font-family:Poppins; text-align:center'>View Product</p></a>"
                    )
                    with st.expander("Scores"):
                        st.html(
                            f"<p style='font-family:Poppins;'>Price Score: {product.price_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>Value Score: {product.value_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>Features Score: {product.features_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>Specs Score: {product.specs_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>Brand Score: {product.brand_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>USP Score: {product.usp_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>Support Score: {product.support_score}/10</p>"
                        )
                        st.html(
                            f"<p style='font-family:Poppins;'>Relevance Score: {product.relevance_score}/10</p>"
                        )
                    st.info(f"OVERALL SCORE: {product.overall_score}/10")
                    st.html(
                        f"<p style='font-family:Poppins; text-align:justify'>{product.score_summary}</p>"
                    )


def display_best_product(best_product: BestProduct):
    col1, _, _ = st.columns(3)
    with col1:
        with st.container(border=True):
            product = best_product.best_product
            container_title(product.recommended_product.name)
            if product.recommended_product.image_url:
                st.image(
                    product.recommended_product.image_url, use_container_width=True
                )
            st.html(
                f"<a href='{product.recommended_product.url}' style='font-family:Poppins; text-align:center'><p style='font-family:Poppins; text-align:center'>View Product</p></a>"
            )
            st.html(
                f"<p style='font-family:Poppins; text-align:justify'>{product.recommended_product.description}</p>"
            )
            st.info(
                f"PRICE: {product.recommended_product.price} {product.recommended_product.currency}"
            )
            with st.expander("Features", expanded=True):
                for feature in product.recommended_product.features:
                    st.html(f"<li style='font-family:Poppins'>{feature}</li>")
            with st.expander("Pros"):
                for pro in product.recommended_product.pros:
                    st.html(f"<li style='font-family:Poppins'>{pro}</li>")
            with st.expander("Cons"):
                for con in product.recommended_product.cons:
                    st.html(f"<li style='font-family:Poppins'>{con}</li>")
            st.info(f"OVERALL SCORE: {product.overall_score}/10")
            st.html(
                f"<p style='font-family:Poppins; text-align:justify'>{best_product.justification}</p>"
            )


agent = ShopWiseAgent

set_page_config(page_title=agent.agent_name)

with st.sidebar:
    # API KEYS

    get_api_key(agent.required_api_keys)

    st.divider()

    # LANGGRAPH WORKFLOW VISUALIZATION

    agent_graph = agent.get_graph()
    add_langgraph_workflow_visualization(agent_graph)

if not keys_missing(agent.required_api_keys):
    if query := st.chat_input(placeholder="Enter your query here..."):
        with st.chat_message("human"):
            st.write(query)

        with st.spinner(text="Searching web for products..."):
            web_search_url = agent_graph.invoke(
                input={"query": query},
                config={"configurable": {"thread_id": "1"}},
            )["web_search_url"]

            st.header("Sources", divider=True)
            st.info(web_search_url)

        with st.spinner(text="Generating list of recommended products..."):
            state = agent_graph.invoke(
                input=None,
                config={"configurable": {"thread_id": "1"}},
            )

            st.header("Recommended Products", divider=True)
            display_recommended_products(
                state["recommended_products"], number_of_products_in_a_row=3
            )

        with st.spinner(
            text="Comparing products and generating list of top 3 products..."
        ):
            state = agent_graph.invoke(
                input=None,
                config={"configurable": {"thread_id": "1"}},
            )

            st.header("Top 3 Products", divider=True)
            display_top3_products(
                top3_products=state["products_comparison"],
                number_of_products_in_a_row=3,
            )

        with st.spinner(text="Getting you the best product..."):
            state = agent_graph.invoke(
                input=None,
                config={"configurable": {"thread_id": "1"}},
            )

            st.header("Best Product", divider=True)
            display_best_product(state["best_product"])

        st.balloons()
