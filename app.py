import streamlit as st
import os

from pages.home import show_home
from pages.eda import show_eda
from pages.segmentation import show_segmentation
from pages.recommendation import show_recommendation
from pages.insights import show_insights
from pages.about import show_about

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------

css_file = os.path.join("assets", "style.css")

if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🛒 Shopper Spectrum")

st.sidebar.markdown(
"""
### Customer Analytics Platform

Machine Learning Powered

✔ Customer Segmentation

✔ Product Recommendation

✔ Business Intelligence
"""
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "👥 Customer Segmentation",
        "🎯 Product Recommendation",
        "📈 Business Insights",
        "ℹ️ About Project"
    ]
)

# ---------------------------------------------------
# PAGE ROUTING
# ---------------------------------------------------

if page == "🏠 Home":
    show_home()

elif page == "📊 EDA Dashboard":
    show_eda()

elif page == "👥 Customer Segmentation":
    show_segmentation()

elif page == "🎯 Product Recommendation":
    show_recommendation()

elif page == "📈 Business Insights":
    show_insights()

elif page == "ℹ️ About Project":
    show_about()
        