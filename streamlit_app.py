import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/main.py", title="Why", icon="📰"),
        st.Page("pages/marketmatrix.py", title="Matrix Chart", icon="📈"),
    ]
)
pg.run()
