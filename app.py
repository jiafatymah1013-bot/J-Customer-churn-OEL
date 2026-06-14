import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction & Segmentation",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")

# ----------------------------------------------------------------------------
# Load saved artifacts (trained in the Jupyter notebook)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")
    results_df = pd.read_csv("model_results.csv", index_col=0)
    data = pd.read_csv("churn_dashboard_data.csv")
    return model, scaler, label_encoders, kmeans, cluster_scaler, results_df, data

model, scaler, label_encoders, kmeans, cluster_scaler, results_df, data = load_artifacts()

FEATURE_ORDER = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
                  'Payment Delay', 'Subscription Type', 'Contract Length',
                  'Total Spend', 'Last Interaction']

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Telecom Churn Analytics")
st.sidebar.markdown(
    """
**Customer Churn Prediction and Customer Segmentation System**

OEL Project — AIC-221L  
Introduction to Machine Learning Lab  
Iqra University, Chak Shahzad Campus
"""
)
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview & EDA", "🤖 Model Comparison", "🔮 Live Prediction", "👥 Customer Segmentation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best Model:** {results_df['Accuracy'].idxmax()}")
st.sidebar.markdown(f"**Dataset size (sample used):** {len(data):,} customers")

# ----------------------------------------------------------------------------
# PAGE 1: Overview & EDA
# ----------------------------------------------------------------------------
if page == "🏠 Overview & EDA":
    st.title("📊 Customer Churn — Overview & Exploratory Data Analysis")
    st.markdown(
        """
        A telecom company is facing a major challenge because many customers are leaving their
        services frequently. This dashboard explores customer behaviour patterns, predicts churn,
        and segments customers for targeted marketing — all based on a real Kaggle Customer Churn
        dataset.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers (sample)", f"{len(data):,}")
    col2.metric("Churn Rate", f"{data['Churn'].mean()*100:.1f}%")
    col3.metric("Avg. Tenure (months)", f"{data['Tenure'].mean():.1f}")
    col4.metric("Avg. Total Spend", f"${data['Total Spend'].mean():.0f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Churn', data=data, palette='Set2', ax=ax)
        ax.set_xticklabels(['Retained', 'Churned'])
        ax.set_title("Customer Churn Distribution")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    with c2:
        st.subheader("Contract Length vs Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        contract_map = {0: 'Annual', 1: 'Monthly', 2: 'Quarterly'}
        plot_data = data.copy()
        plot_data['Contract Length'] = plot_data['Contract Length'].map(contract_map)
        plot_data['Churn Status'] = plot_data['Churn'].map({0: 'Retained', 1: 'Churned'})
        sns.countplot(x='Contract Length', hue='Churn Status', data=plot_data, palette='Set1', ax=ax)
        ax.set_title("Contract Length vs Churn")
        ax.set_xlabel("Contract Length")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_data = data.copy()
        plot_data['Churn Status'] = plot_data['Churn'].map({0: 'Retained', 1: 'Churned'})
        sns.boxplot(x='Churn Status', y='Tenure', data=plot_data, palette='Set3', ax=ax)
        ax.set_title("Tenure Distribution by Churn Status")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Tenure (months)")
        st.pyplot(fig)

    with c4:
        st.subheader("Average Support Calls by Churn Status")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_data = data.copy()
        plot_data['Churn Status'] = plot_data['Churn'].map({0: 'Retained', 1: 'Churned'})
        sns.barplot(x='Churn Status', y='Support Calls', data=plot_data, palette='magma',
                    estimator=np.mean, ax=ax)
        ax.set_title("Average Support Calls")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Average Support Calls")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = data.drop(columns=['KMeans_Cluster']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Business Insights")
    st.markdown(
        """
        - **Support Calls:** Customers who churn make noticeably more support calls on average —
          unresolved issues are a major churn driver.
        - **Contract Length:** Monthly-contract customers churn far more often than annual-contract
          customers. Encouraging longer contracts improves retention.
        - **Tenure:** Newer customers (low tenure) are more likely to churn — onboarding experience
          in the first few months is critical.
        - **Payment Delay:** Higher payment delays correlate with higher churn — an early-warning
          signal for the retention team.
        """
    )

# ----------------------------------------------------------------------------
# PAGE 2: Model Comparison
# ----------------------------------------------------------------------------
elif page == "🤖 Model Comparison":
    st.title("🤖 Supervised Model Comparison")
    st.markdown(
        "Five classification algorithms were trained and evaluated on the churn dataset: "
        "**Decision Tree, Random Forest, KNN, Logistic Regression, and Naive Bayes**."
    )

    st.subheader("Performance Metrics")
    st.dataframe(results_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"))

    st.subheader("Metric Comparison Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title("Supervised Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Metric", loc="lower right")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    best_model_name = results_df['Accuracy'].idxmax()
    st.success(
        f"🏆 **Best Performing Model: {best_model_name}** "
        f"(Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.2%}, "
        f"F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.2%})"
    )

    st.markdown(
        f"""
        ### Why {best_model_name}?
        {best_model_name} achieved the highest combination of Accuracy and F1-Score among all
        tested models. A high F1-Score balances Precision (avoiding false churn alarms) and Recall
        (catching real churners) — both essential for an effective retention strategy.
        """
    )

# ----------------------------------------------------------------------------
# PAGE 3: Live Prediction
# ----------------------------------------------------------------------------
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Enter a customer's details below to predict whether they are likely to churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 65, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        tenure = st.slider("Tenure (months)", 0, 60, 24)

    with col2:
        usage_freq = st.slider("Usage Frequency (per month)", 0, 30, 15)
        support_calls = st.slider("Support Calls", 0, 10, 3)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 10)

    with col3:
        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.slider("Total Spend ($)", 0, 1000, 500)
        last_interaction = st.slider("Days Since Last Interaction", 1, 30, 14)

    if st.button("Predict Churn", type="primary"):
        # Encode categorical inputs using the same encoders from training
        gender_enc = label_encoders['Gender'].transform([gender])[0]
        sub_enc = label_encoders['Subscription Type'].transform([subscription])[0]
        contract_enc = label_encoders['Contract Length'].transform([contract])[0]

        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender_enc,
            'Tenure': tenure,
            'Usage Frequency': usage_freq,
            'Support Calls': support_calls,
            'Payment Delay': payment_delay,
            'Subscription Type': sub_enc,
            'Contract Length': contract_enc,
            'Total Spend': total_spend,
            'Last Interaction': last_interaction
        }])[FEATURE_ORDER]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **Prediction: This customer is LIKELY TO CHURN** "
                     f"(Confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ **Prediction: This customer is LIKELY TO STAY** "
                       f"(Confidence: {probability[0]*100:.1f}%)")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(['Probability'], [probability[1]], color='salmon', label='Churn')
        ax.barh(['Probability'], [probability[0]], left=[probability[1]], color='lightgreen', label='Retain')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)
        ax.set_title("Churn vs Retention Probability")
        st.pyplot(fig)

        st.markdown("### Recommendation")
        if prediction == 1:
            st.markdown(
                """
                - Reach out proactively with a retention offer.
                - If on a monthly contract, offer a discount to switch to a quarterly/annual plan.
                - Check recent support tickets — high support call volume often signals frustration.
                """
            )
        else:
            st.markdown("- This customer shows healthy engagement. Continue standard service quality.")

# ----------------------------------------------------------------------------
# PAGE 4: Customer Segmentation
# ----------------------------------------------------------------------------
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation (Clustering)")
    st.markdown(
        "Customers were grouped into segments using **K-Means Clustering** based on usage "
        "patterns and behaviour (Churn label was excluded from clustering)."
    )

    st.subheader("Cluster Sizes")
    cluster_counts = data['KMeans_Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts.plot(kind='bar', color='teal', ax=ax)
    ax.set_title("Number of Customers per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Cluster Profiles (Average Feature Values)")
    profile_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay',
                     'Total Spend', 'Last Interaction', 'Churn']
    cluster_profile = data.groupby('KMeans_Cluster')[profile_cols].mean().round(2)
    st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', axis=0))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Total Spend per Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        cluster_profile['Total Spend'].plot(kind='bar', color='teal', ax=ax)
        ax.set_title("Average Total Spend per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Average Total Spend")
        st.pyplot(fig)

    with c2:
        st.subheader("Average Churn Rate per Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        cluster_profile['Churn'].plot(kind='bar', color='indianred', ax=ax)
        ax.set_title("Average Churn Rate per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Average Churn Rate")
        st.pyplot(fig)

    st.subheader("Cluster Visualization (PCA Projection)")
    from sklearn.decomposition import PCA
    X_cluster = data[FEATURE_ORDER]
    X_scaled = cluster_scaler.transform(X_cluster)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=data['KMeans_Cluster'], cmap='viridis', alpha=0.5, s=10)
    ax.set_title("Customer Segments (K-Means, PCA Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Segment Interpretation & Marketing Recommendations")
    st.markdown(
        """
        - **High-spend, low-churn segments** represent loyal customers — reward them with loyalty
          perks to maintain satisfaction.
        - **Low-tenure, high-support-call segments** are at-risk customers — prioritize proactive
          retention outreach and improved onboarding.
        - **High payment-delay segments** may need flexible billing options or reminders to reduce
          churn risk linked to payment friction.
        - Use these segments to design targeted marketing campaigns rather than a one-size-fits-all
          approach.
        """
    )

st.markdown("---")
st.caption("OEL Project — AIC-221L Introduction to Machine Learning Lab | Iqra University, Chak Shahzad Campus, Islamabad")
