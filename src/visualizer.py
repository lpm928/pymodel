import altair as alt
import pandas as pd
import streamlit as st

def plot_classification_metrics(report_dict):
    """Plot precision/recall/f1-score from report dict."""
    # Convert '0', '1', 'macro avg' etc to dataframe
    data = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            metrics['class'] = label
            data.append(metrics)
    
    df_metrics = pd.DataFrame(data)
    
    # Filter out accuracy row if it's there as a confusing dict
    df_metrics = df_metrics[df_metrics['class'] != 'accuracy']
    
    # Melt for altair
    df_melt = df_metrics.melt(id_vars=['class'], value_vars=['precision', 'recall', 'f1-score'], 
                              var_name='metric', value_name='score')
    
    chart = alt.Chart(df_melt).mark_bar().encode(
        x='class:N',
        y='score:Q',
        color='metric:N',
        column='metric:N'
    ).properties(title="Model Performance Metrics")
    
    st.altair_chart(chart, use_container_width=True)

def plot_feature_importance(model, feature_names):
    """Plot feature importance if model supports it."""
    if hasattr(model, 'feature_importances_'):
        df_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).head(10)
        
        chart = alt.Chart(df_imp).mark_bar().encode(
            x=alt.X('importance:Q', title='Importance'),
            y=alt.Y('feature:N', sort='-x', title='Feature'),
            color=alt.value('#4c78a8')
        ).properties(title="Top 10 Feature Importance")
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("This model type does not support simple feature importance.")

def plot_clusters_2d(df, cluster_col, valid_cols):
    """
    Plot 2D scatter of clusters using the first two numerical columns available.
    valid_cols: list of numerical column names in df.
    """
    if len(valid_cols) < 2:
        st.warning("Not enough numerical features to plot clusters (Need >= 2).")
        return
        
    c1, c2 = valid_cols[0], valid_cols[1]
    
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(c1, scale=alt.Scale(zero=False)),
        y=alt.Y(c2, scale=alt.Scale(zero=False)),
         color=alt.Color(f"{cluster_col}:N", title="Cluster"),
        tooltip=[c1, c2, cluster_col]
    ).interactive().properties(title=f"Cluster Visualization ({c1} vs {c2})")
    
    st.altair_chart(chart, use_container_width=True)

def plot_funnel(stages_dict):
    """
    Plot a Funnel Chart.
    stages_dict: {'Stage Name': Count, ...}
    """
    if not stages_dict:
        return
        
    data = pd.DataFrame(list(stages_dict.items()), columns=['Stage', 'Count'])
    
    # Calculate drop-off or just simple bars
    # Using bar chart that looks like funnel (sorted descending)
    
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Count:Q', stack=None),
        y=alt.Y('Stage:N', sort=None), # Assume inputs are ordered
        color=alt.Color('Stage:N', legend=None),
        tooltip=['Stage', 'Count']
    ).properties(title="Conversion Funnel")
    
    text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text='Count:Q'
    )
    
    st.altair_chart(chart + text, use_container_width=True)

def plot_accuracy_trend(history_df):
    """
    Plot Model Accuracy Trend over versions.
    history_df expects columns: ['Version', 'Metric', 'Score']
    """
    if history_df.empty:
        return

    chart = alt.Chart(history_df).mark_line(point=True).encode(
        x='Version:N', # Nominal to keep discrete steps
        y='Score:Q',
        color='Metric:N',
        tooltip=['Version', 'Metric', 'Score']
    ).properties(title="Model Accuracy Trend")
    
    st.altair_chart(chart, use_container_width=True)

def explain_feature_importance(model, feature_names):
    """
    Generate a text summary of feature importance.
    Returns: list of strings (bullet points).
    """
    if not hasattr(model, 'feature_importances_'):
        return ["此模型不支援特徵權重分析 (No feature importance available)."]
        
    importances = model.feature_importances_
    # Create DataFrame
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # Generate insights
    top_3 = df_imp.head(3)
    total_imp = df_imp['importance'].sum()
    
    insights = []
    insights.append(f"**模型依賴度最高的前 3 個特徵 (Top 3 Drivers):**")
    
    for i, row in top_3.iterrows():
        pct = (row['importance'] / total_imp) * 100
        insights.append(f"- **{row['feature']}**: 影響力約 **{pct:.1f}%**")
        
    # Analyze long tail?
    if len(df_imp) > 5:
        tail_sum = df_imp.iloc[5:]['importance'].sum()
        if tail_sum < 0.1:
            insights.append("💡 提示: 其他特徵影響力較低，模型主要由前幾大特徵主導。")
    
    return insights
