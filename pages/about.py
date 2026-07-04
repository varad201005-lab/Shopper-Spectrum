import streamlit as st


def show_about():

    st.title("ℹ️ About Shopper Spectrum")

    st.markdown("""
# 🛒 Shopper Spectrum

### Customer Segmentation & Product Recommendation in E-Commerce

Shopper Spectrum is an end-to-end Data Analytics and Machine Learning project
developed to analyze customer purchasing behaviour in an online retail business.

The application helps businesses understand customer purchasing patterns,
identify valuable customer segments,
and recommend relevant products using collaborative filtering.
""")

    st.markdown("---")

    # ==========================================================
    # BUSINESS OBJECTIVE
    # ==========================================================

    st.header("🎯 Business Objective")

    st.markdown("""
The primary objective of this project is to help an e-commerce company:

- Understand customer buying behaviour.
- Identify high-value customers.
- Improve customer retention.
- Increase revenue through targeted marketing.
- Recommend similar products.
- Support data-driven business decisions.
""")

    st.markdown("---")

    # ==========================================================
    # PROJECT WORKFLOW
    # ==========================================================

    st.header("⚙️ Project Workflow")

    st.markdown("""
1. Dataset Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering (RFM Analysis)
5. Customer Segmentation using KMeans Clustering
6. Product Recommendation using Item-Based Collaborative Filtering
7. Business Insights
8. Streamlit Dashboard Deployment
""")

    st.markdown("---")

    # ==========================================================
    # TECHNOLOGIES
    # ==========================================================

    st.header("💻 Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
### Programming

- Python
- Pandas
- NumPy
- Streamlit

### Visualization

- Plotly
- Matplotlib
- Seaborn
""")

    with col2:

        st.markdown("""
### Machine Learning

- Scikit-Learn
- KMeans Clustering
- StandardScaler
- Cosine Similarity

### Model Storage

- Joblib
""")

    st.markdown("---")

    # ==========================================================
    # DATASET
    # ==========================================================

    st.header("📂 Dataset")

    st.markdown("""
The dataset contains online retail transaction records.

### Features

- Invoice Number
- Product Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

The cleaned dataset is used for customer segmentation,
recommendation modelling,
and business intelligence.
""")

    st.markdown("---")

    # ==========================================================
    # MACHINE LEARNING
    # ==========================================================

    st.header("🤖 Machine Learning Techniques")

    st.markdown("""
### Customer Segmentation

- RFM Analysis
- Feature Scaling
- KMeans Clustering
- Elbow Method
- Silhouette Score

### Product Recommendation

- Customer × Product Matrix
- Cosine Similarity
- Item-Based Collaborative Filtering
""")

    st.markdown("---")

    # ==========================================================
    # FEATURES
    # ==========================================================

    st.header("✨ Dashboard Features")

    st.markdown("""
✅ Interactive Dashboard

✅ Customer Segmentation

✅ Product Recommendation

✅ Business Insights

✅ Download Customer Segmentation

✅ Download Cleaned Dataset

✅ Interactive Charts

✅ Executive Reports
""")

    st.markdown("---")

    # ==========================================================
    # FUTURE ENHANCEMENTS
    # ==========================================================

    st.header("🚀 Future Enhancements")

    st.markdown("""
- Real-time Recommendation Engine
- Customer Lifetime Value Prediction
- Deep Learning Recommendation Models
- Sales Forecasting
- AI Chatbot Integration
- Cloud Deployment
""")

    st.markdown("---")

    # ==========================================================
    # DEVELOPER
    # ==========================================================

    st.header("👨‍💻 Developed By")

    st.success("""
**Varad Kulkarni** 
""")

    st.markdown("---")

    st.info("""
Thank you for exploring **Shopper Spectrum**.

This project demonstrates how Data Analytics,
Machine Learning,
and Business Intelligence can work together
to improve customer experience and business decision-making.
""")
    