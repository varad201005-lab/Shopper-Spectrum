import streamlit as st

from utils import (
    load_models,
    recommend_product
)


def show_recommendation():

    st.title("🎯 Product Recommendation System")

    st.markdown("""
Recommend similar products using **Item-Based Collaborative Filtering**.
""")

    st.markdown("---")

    # -------------------------------------------------------
    # Load Models
    # -------------------------------------------------------

    _, _, similarity_df, product_list = load_models()

    # -------------------------------------------------------
    # Product Selection
    # -------------------------------------------------------

    st.subheader("🔍 Search Product")

    product = st.selectbox(
        "Select Product",
        sorted(product_list),
        index=None,
        placeholder="Type or select a product..."
    )

    if st.button("Get Recommendations"):

        if product is None:

            st.warning("Please select a product.")

        else:

            recommendations = recommend_product(
                product,
                similarity_df
            )

            if recommendations is None:

                st.error("Product not found.")

            else:

                st.success(
                    f"Top 5 Recommendations for: **{product}**"
                )

                st.markdown("---")

                for i, item in enumerate(recommendations, start=1):

                    with st.container(border=True):

                        st.markdown(
                            f"### 🛍️ Recommendation {i}"
                        )

                        st.markdown(
                            f"**Product:** {item}"
                        )

                        st.write(
                            f"Customers who purchased **{product}** also frequently purchased **{item}**."
                        )

    st.markdown("---")

    # -------------------------------------------------------
    # About Recommendation System
    # -------------------------------------------------------

    st.subheader("ℹ️ How Recommendations Work")

    st.info("""
This recommendation engine uses **Item-Based Collaborative Filtering**.

It identifies products that are commonly purchased together by customers,
computes their similarity using **Cosine Similarity**, and returns the
top five most similar products.

This approach helps improve:
- Cross-selling opportunities
- Customer shopping experience
- Product discovery
- Sales and revenue
""")
    