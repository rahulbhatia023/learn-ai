import streamlit as st

from agents.shopwise_agent import ShopWiseAgent, RecommendedProduct
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
                    st.html(
                        f"<p style='font-family:Poppins;'>Price: {recommended_product.price} {recommended_product.currency}</p>"
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
            agent_graph.invoke(
                input={"query": query},
                config={"configurable": {"thread_id": "1"}},
            )

        with st.spinner(text="Generating list of recommended products..."):
            state = agent_graph.invoke(
                input=None,
                config={"configurable": {"thread_id": "1"}},
            )
            display_recommended_products(
                state["recommended_products"], number_of_products_in_a_row=3
            )
