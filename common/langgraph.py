import streamlit as st


def add_langgraph_workflow_visualization(graph):
    st.html(
        "<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>LangGraph Workflow Visualization</h3>"
    )

    st.html(
        """
        <style>
            [data-testid="stImage"] {
                border-radius: 10px;
                overflow: hidden;
            }
        </style>
        """
    )

    st.image(
        graph.get_graph(xray=1).draw_mermaid_png(),
        use_container_width=True,
    )
