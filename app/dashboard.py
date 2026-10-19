import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

REP = Path("reports")

st.set_page_config(
    page_title="RetailPulse",
    page_icon="cart",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        padding: 8px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 600;
        color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5;
        color: white !important;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
    }
    h2 {
        color: #334155;
        font-weight: 600;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# RetailPulse")
st.markdown("### AI-Powered Customer Analytics & Demand Forecasting Platform")
st.markdown("---")

with st.sidebar:
    st.title("RetailPulse")
    st.markdown("AI Retail Analytics")
    st.markdown("---")
    st.markdown("**Dataset:** Online Retail II")
    st.markdown("**Records:** 1,067,371")
    st.markdown("**Period:** Dec 2009 - Dec 2011")
    st.markdown("---")
    st.markdown("**Tabs:**")
    st.markdown("- Forecast")
    st.markdown("- Segments")
    st.markdown("- Churn")
    st.markdown("- Inventory")
    st.markdown("---")
    st.caption("Built with Streamlit + Prophet + Scikit-learn")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Segments", "Churn", "Inventory"])

with tab1:
    st.header("Demand Forecast")
    st.caption("Model comparison on last 30 days holdout")

    f = REP / "forecast_total_metrics.csv"
    if f.exists():
        metrics = pd.read_csv(f)
        cols = st.columns(len(metrics))
        for i, (_, row) in enumerate(metrics.iterrows()):
            with cols[i]:
                st.metric(
                    label=row["Model"],
                    value=f"MAE {row['MAE']:,.0f}",
                    delta=f"RMSE {row['RMSE']:,.0f}",
                    delta_color="off",
                )

        st.markdown("---")
        st.subheader("Detailed Metrics")
        st.dataframe(metrics, use_container_width=True)

        img_path = REP / "forecast_total.png"
        if img_path.exists():
            st.subheader("Forecast Visualization")
            st.image(str(img_path), use_container_width=True)
    else:
        st.warning("Run forecast pipeline first.")

with tab2:
    st.header("Customer Segments")
    st.caption("RFM analysis + K-Means clustering")

    f = REP / "customer_segments.csv"
    if f.exists():
        seg = pd.read_csv(f)

        col1, col2 = st.columns([1, 2])

        with col1:
            counts = seg["Segment"].value_counts().reset_index()
            counts.columns = ["Segment", "Count"]
            fig = px.pie(
                counts,
                names="Segment",
                values="Count",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Customer Distribution",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                seg,
                x="Recency",
                y="Monetary",
                color="Segment",
                size="Frequency",
                hover_data=["CustomerID"],
                title="Recency vs Monetary",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Segment Profiles")
        profiles_path = REP / "segment_profiles.csv"
        if profiles_path.exists():
            st.dataframe(pd.read_csv(profiles_path), use_container_width=True)

        st.markdown("---")
        st.subheader("Customer Data (first 100)")
        st.dataframe(seg.head(100), use_container_width=True)
    else:
        st.warning("Run segmentation first.")

with tab3:
    st.header("Churn Risk Analysis")
    st.caption("Churn = no purchase in 60 days")

    f = REP / "churn_risk_list.csv"
    if f.exists():
        churn = pd.read_csv(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(churn):,}")
        with col2:
            churned = int(churn["Churned"].sum())
            st.metric("Churned", f"{churned:,}", f"{churned/len(churn)*100:.0f}%")
        with col3:
            high_risk = int((churn["ChurnProbability"] > 0.5).sum())
            st.metric("High Risk", f"{high_risk:,}")

        st.markdown("---")
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(
            churn,
            x="ChurnProbability",
            nbins=40,
            color_discrete_sequence=["#ef4444"],
            title="Distribution of churn probabilities",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        threshold = st.slider("Filter by churn probability", 0.0, 1.0, 0.5, 0.05)
        filtered = churn[churn["ChurnProbability"] > threshold]
        st.info(f"{len(filtered)} customers above {threshold:.0%} churn probability")

        st.dataframe(
            filtered[["CustomerID", "TotalOrders", "TotalSpend", "RecencyDays", "ChurnProbability"]].head(100),
            use_container_width=True,
        )
    else:
        st.warning("Run churn model first.")

with tab4:
    st.header("Inventory Recommendations")
    st.caption("Reorder points based on 7-day demand forecast")

    f = REP / "inventory_recommendations.csv"
    if f.exists():
        inv = pd.read_csv(f)
        reorder = inv[inv["Action"] == "REORDER"]
        ok = inv[inv["Action"] == "OK"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total SKUs", len(inv))
        with col2:
            st.metric("Reorder Needed", len(reorder), f"{len(reorder)/len(inv)*100:.0f}%")
        with col3:
            st.metric("Stock OK", len(ok))

        st.markdown("---")

        if len(reorder) > 0:
            st.error(f"{len(reorder)} SKUs need immediate reorder")

        fig = px.bar(
            inv.head(20),
            x="StockCode",
            y=["Forecast7d", "ReorderPoint"],
            barmode="group",
            title="Top 20 SKUs: Forecast vs Reorder Point",
            color_discrete_sequence=["#4f46e5", "#ef4444"],
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Full Recommendations")
        st.dataframe(inv, use_container_width=True)

        csv = inv.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="inventory_recommendations.csv",
            mime="text/csv",
        )
    else:
        st.warning("Run inventory optimization first.")

st.markdown("---")
st.caption("RetailPulse - AI-Powered Retail Analytics")
