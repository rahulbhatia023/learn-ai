import streamlit as st

from agents.shopwise_agent import ShopWiseAgent
from common.langgraph import add_langgraph_workflow_visualization
from common.page import get_api_key, keys_missing
from common.theme import set_page_config, container_title

agent = ShopWiseAgent

set_page_config(page_title=agent.agent_name)


def display_products(products: list, number_of_products_in_a_row: int):
    for i in range(0, len(products), number_of_products_in_a_row):
        products_in_a_row = products[i : i + number_of_products_in_a_row]
        cols = st.columns(number_of_products_in_a_row)
        for j in range(0, len(products_in_a_row)):
            with cols[j]:
                product = products_in_a_row[j]
                with st.container(border=True):
                    container_title(product["title"])
                    st.html(
                        f"<p style='font-family:Poppins; text-align: justify'>{product["content"]}</p>"
                    )
                    with st.expander("Pros"):
                        for pro in product["pros"]:
                            st.html(f"<li style='font-family:Poppins'>{pro}</li>")
                    with st.expander("Cons"):
                        for con in product["cons"]:
                            st.html(f"<li style='font-family:Poppins'>{con}</li>")


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

        with st.spinner(text="Fetching top results..."):
            result = agent_graph.invoke(
                input={"query": query},
                config={"configurable": {"thread_id": "1"}},
            )
            products = result["product_schema"]
            display_products(products, 2)

        with st.spinner("Comparing and reviewing the products.."):
            result = agent_graph.invoke(
                input=None,
                config={"configurable": {"thread_id": "1"}},
            )
            st.write(result)
