import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def generate_plot(llm_output, current_data, column_mapping=None):
    """
    Generate a plotly chart based on LLM output and return both the chart object and response text.

    Args:
        llm_output (dict): Dictionary with plot specifications from LLM
        current_data (pd.DataFrame): The DataFrame to plot
        column_mapping (dict): Optional mapping from standardized column names to original ones

    Returns:
        tuple: (chart, response_text) where chart is a plotly figure or None if error
    """
    try:
        plot_type = llm_output.get("plot_type")
        x_col = llm_output.get("x_column")
        y_col = llm_output.get("y_column")
        color_col = llm_output.get("color_column")
        names_col = llm_output.get("names_column")
        title = llm_output.get("title", f"Plot based on user query")

        # Validate columns exist in the DataFrame
        all_cols = list(current_data.columns)

        def validate_col(col_name):
            return col_name if col_name is not None and col_name in all_cols else None

        x_col_valid = validate_col(x_col)
        y_col_valid = validate_col(y_col)
        color_col_valid = validate_col(color_col)
        names_col_valid = validate_col(names_col)

        def get_original_name(col):
            if column_mapping and col in column_mapping:
                return column_mapping.get(col, col)
            return col

        chart = None
        response_text = "I've generated a visualization for you."

        if plot_type == 'pie':
            if not names_col_valid:
                return None, "Cannot create pie chart: missing column for names."
            if not pd.api.types.is_string_dtype(
                    current_data[names_col_valid]) and not pd.api.types.is_categorical_dtype(
                current_data[names_col_valid]):
                return None, f"Cannot create pie chart: '{get_original_name(names_col_valid)}' is not a categorical column."
            unique_count = current_data[names_col_valid].nunique()
            if unique_count > 25:
                return None, f"Too many unique values ({unique_count}) in '{get_original_name(names_col_valid)}' for a pie chart. Try a bar chart instead."
            counts = current_data[names_col_valid].value_counts().reset_index()
            counts.columns = [names_col_valid, 'count']
            chart = px.pie(counts, names=names_col_valid, values='count', title=title, hole=0.3)
            response_text = f"Generated pie chart showing distribution of '{get_original_name(names_col_valid)}'."

        elif plot_type == 'bar':
            if not x_col_valid:
                return None, "Cannot create bar chart: missing x-axis column."
            if pd.api.types.is_numeric_dtype(current_data[x_col_valid]):
                return None, f"Bar chart x-axis ('{get_original_name(x_col_valid)}') is numeric. Consider using a histogram instead."
            counts = current_data[x_col_valid].value_counts().reset_index()
            counts.columns = [x_col_valid, 'count']
            chart = px.bar(counts, x=x_col_valid, y='count', color=color_col_valid, title=title)
            response_text = f"Generated bar chart showing counts for '{get_original_name(x_col_valid)}'."
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."

        elif plot_type == 'histogram':
            if not x_col_valid:
                return None, "Cannot create histogram: missing x-axis column."
            if not pd.api.types.is_numeric_dtype(
                    current_data[x_col_valid]) and not pd.api.types.is_datetime64_any_dtype(current_data[x_col_valid]):
                return None, f"Histogram requires a numeric or date column. '{get_original_name(x_col_valid)}' is not. Try a bar chart instead."
            chart = px.histogram(current_data, x=x_col_valid, color=color_col_valid, title=title)
            response_text = f"Generated histogram for '{get_original_name(x_col_valid)}'."
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."

        elif plot_type == 'box':
            if not y_col_valid:
                return None, "Cannot create box plot: missing y-axis column (numeric)."
            if not pd.api.types.is_numeric_dtype(current_data[y_col_valid]):
                return None, f"Box plot requires a numeric y-axis. '{get_original_name(y_col_valid)}' is not numeric."
            chart = px.box(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                           points="outliers")
            response_text = f"Generated box plot for '{get_original_name(y_col_valid)}'."
            if x_col_valid:
                response_text += f" grouped by '{get_original_name(x_col_valid)}'"
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."

        elif plot_type == 'scatter':
            if not x_col_valid or not y_col_valid:
                return None, f"Cannot create scatter plot: need both x and y columns."
            if not (pd.api.types.is_numeric_dtype(current_data[x_col_valid]) or pd.api.types.is_datetime64_any_dtype(
                    current_data[x_col_valid])):
                return None, f"Scatter plot x-axis ('{get_original_name(x_col_valid)}') must be numeric or date."
            if not (pd.api.types.is_numeric_dtype(current_data[y_col_valid]) or pd.api.types.is_datetime64_any_dtype(
                    current_data[y_col_valid])):
                return None, f"Scatter plot y-axis ('{get_original_name(y_col_valid)}') must be numeric or date."
            chart = px.scatter(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                               hover_data=current_data.columns)
            response_text = f"Generated scatter plot of '{get_original_name(x_col_valid)}' vs '{get_original_name(y_col_valid)}'."
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."
        else:
            return None, f"Unsupported plot type: {plot_type}. Please try a different visualization."

        return chart, response_text

    except Exception as e:
        return None, f"Error generating visualization: {e}"


def create_insights_chart(data, labels=None, values=None, chart_type='pie', title=None):
    """
    Create a simple chart for analytics insights.
    """
    try:
        if chart_type == 'pie' and labels:
            counts = data[labels].value_counts().reset_index()
            counts.columns = [labels, 'count']
            fig = px.pie(counts, names=labels, values='count', title=title)
            return fig
        elif chart_type == 'bar' and labels:
            if values:
                fig = px.bar(data, x=labels, y=values, title=title)
            else:
                counts = data[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                fig = px.bar(counts, x=labels, y='count', title=title)
            return fig
        elif chart_type == 'histogram' and labels:
            fig = px.histogram(data, x=labels, title=title)
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None


def auto_visualize(results_df: pd.DataFrame):
    """
    Automatically generates a set of relevant visualizations based on the DataFrame's structure.
    This function is a powerful tool for quick data exploration.
    """
    if results_df is None or results_df.empty:
        st.info("💡 No data available for visualization.")
        return

    st.subheader("📊 Automatic Visualizations & Insights")

    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = results_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # Filter out high-cardinality categorical columns to avoid cluttered charts
    filtered_categorical_cols = [
        col for col in categorical_cols
        if 1 < results_df[col].nunique() <= 30
    ]

    charts_created = 0
    max_charts = 4

    # Time series analysis
    if date_cols and numeric_cols and charts_created < max_charts:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        try:
            st.write(f"**Time Series Analysis: `{num_col}` over `{date_col}`**")
            chart = px.line(results_df, x=date_col, y=num_col, title=f"Trend of {num_col} over time", markers=True)
            st.plotly_chart(chart, use_container_width=True)
            charts_created += 1
        except Exception as e:
            st.warning(f"Could not create time series plot: {e}")

    # Categorical distribution (Pie or Bar)
    if filtered_categorical_cols and charts_created < max_charts:
        cat_col = filtered_categorical_cols[0]
        try:
            if results_df[cat_col].nunique() <= 10:
                st.write(f"**Distribution by `{cat_col}`**")
                chart = create_insights_chart(results_df, labels=cat_col, chart_type='pie',
                                              title=f"Distribution of {cat_col}")
                st.plotly_chart(chart, use_container_width=True)
                charts_created += 1
            else:
                st.write(f"**Count by `{cat_col}`**")
                chart = create_insights_chart(results_df, labels=cat_col, chart_type='bar', title=f"Count by {cat_col}")
                st.plotly_chart(chart, use_container_width=True)
                charts_created += 1
        except Exception as e:
            st.warning(f"Could not create categorical plot: {e}")

    # Numeric distribution (Histogram)
    if numeric_cols and charts_created < max_charts:
        num_col = numeric_cols[0]
        try:
            st.write(f"**Distribution of `{num_col}`**")
            chart = create_insights_chart(results_df, labels=num_col, chart_type='histogram',
                                          title=f"Distribution of {num_col}")
            st.plotly_chart(chart, use_container_width=True)
            charts_created += 1
        except Exception as e:
            st.warning(f"Could not create histogram: {e}")

    # Scatter plot for relationship between two numeric variables
    if len(numeric_cols) >= 2 and charts_created < max_charts:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        try:
            st.write(f"**Relationship between `{x_col}` and `{y_col}`**")
            chart = px.scatter(results_df, x=x_col, y=y_col, title=f"{x_col} vs. {y_col}",
                               hover_data=results_df.columns)
            st.plotly_chart(chart, use_container_width=True)
            charts_created += 1
        except Exception as e:
            st.warning(f"Could not create scatter plot: {e}")

    if charts_created == 0:
        st.info(
            "Could not automatically generate visualizations for this query result. The data structure may be too complex or not suitable for standard charts.")

    # Always show summary statistics for numeric columns
    if numeric_cols:
        st.write("**Summary Statistics**")
        st.dataframe(results_df[numeric_cols].describe(), use_container_width=True)