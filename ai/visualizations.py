import pandas as pd
import plotly.express as px
import streamlit as st


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

        # Get original column names if mapping is available
        def get_original_name(col):
            if column_mapping and col in column_mapping:
                return column_mapping.get(col, col)
            return col

        # Generate chart based on plot type
        chart = None
        response_text = "I've generated a visualization for you."

        if plot_type == 'pie':
            if not names_col_valid:
                return None, "Cannot create pie chart: missing column for names."

            # Ensure the column is suitable for pie (categorical with limited unique values)
            if not pd.api.types.is_string_dtype(
                    current_data[names_col_valid]) and not pd.api.types.is_categorical_dtype(
                    current_data[names_col_valid]):
                return None, f"Cannot create pie chart: '{get_original_name(names_col_valid)}' is not a categorical column."

            unique_count = current_data[names_col_valid].nunique()
            if unique_count > 25:
                return None, f"Too many unique values ({unique_count}) in '{get_original_name(names_col_valid)}' for a pie chart. Try a bar chart instead."

            # Calculate counts for the pie chart
            counts = current_data[names_col_valid].value_counts().reset_index()
            counts.columns = [names_col_valid, 'count']  # Rename columns for px.pie
            chart = px.pie(counts, names=names_col_valid, values='count', title=title, hole=0.3)
            response_text = f"Generated pie chart showing distribution of '{get_original_name(names_col_valid)}'."

        elif plot_type == 'bar':
            if not x_col_valid:
                return None, "Cannot create bar chart: missing x-axis column."

            # Bar charts typically show counts per category or sum of y per category
            if pd.api.types.is_numeric_dtype(current_data[x_col_valid]):
                return None, f"Bar chart x-axis ('{get_original_name(x_col_valid)}') is numeric. Consider using a histogram instead."

            # Calculate counts per category
            counts = current_data[x_col_valid].value_counts().reset_index()
            counts.columns = [x_col_valid, 'count']
            chart = px.bar(counts, x=x_col_valid, y='count', color=color_col_valid, title=title)

            response_text = f"Generated bar chart showing counts for '{get_original_name(x_col_valid)}'"
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."
            else:
                response_text += "."

        elif plot_type == 'histogram':
            if not x_col_valid:
                return None, "Cannot create histogram: missing x-axis column."

            if not pd.api.types.is_numeric_dtype(
                    current_data[x_col_valid]) and not pd.api.types.is_datetime64_any_dtype(current_data[x_col_valid]):
                return None, f"Histogram requires a numeric or date column. '{get_original_name(x_col_valid)}' is not. Try a bar chart instead."

            chart = px.histogram(current_data, x=x_col_valid, color=color_col_valid, title=title)

            response_text = f"Generated histogram for '{get_original_name(x_col_valid)}'"
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."
            else:
                response_text += "."

        elif plot_type == 'box':
            if not y_col_valid:
                return None, "Cannot create box plot: missing y-axis column (numeric)."

            if not pd.api.types.is_numeric_dtype(current_data[y_col_valid]):
                return None, f"Box plot requires a numeric y-axis. '{get_original_name(y_col_valid)}' is not numeric."

            chart = px.box(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                           points="outliers")

            response_text = f"Generated box plot for '{get_original_name(y_col_valid)}'"
            if x_col_valid:
                response_text += f" grouped by '{get_original_name(x_col_valid)}'"
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."
            else:
                response_text += "."

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

            response_text = f"Generated scatter plot of '{get_original_name(x_col_valid)}' vs '{get_original_name(y_col_valid)}'"
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."
            else:
                response_text += "."
        else:
            return None, f"Unsupported plot type: {plot_type}. Please try a different visualization."

        # Add summary stats if plot is successful
        if chart:
            primary_plot_col = names_col_valid or x_col_valid or y_col_valid
            if primary_plot_col and primary_plot_col in current_data.columns:
                temp_data_for_stats = current_data[primary_plot_col].dropna()
                summary_text = ""

                if pd.api.types.is_numeric_dtype(temp_data_for_stats) and not temp_data_for_stats.empty:
                    desc = temp_data_for_stats.describe()
                    summary_text = (f"\n\n**Summary for '{get_original_name(primary_plot_col)}':** "
                                    f"Mean: {desc.get('mean', float('nan')):.2f}, "
                                    f"Std: {desc.get('std', float('nan')):.2f}, "
                                    f"Min: {desc.get('min', float('nan')):.2f}, "
                                    f"Max: {desc.get('max', float('nan')):.2f}, "
                                    f"Count: {int(desc.get('count', 0))}")

                elif pd.api.types.is_datetime64_any_dtype(temp_data_for_stats) and not temp_data_for_stats.empty:
                    summary_text = (f"\n\n**Summary for '{get_original_name(primary_plot_col)}':** "
                                    f"Earliest: {temp_data_for_stats.min().strftime('%Y-%m-%d')}, "
                                    f"Latest: {temp_data_for_stats.max().strftime('%Y-%m-%d')}, "
                                    f"Count: {len(temp_data_for_stats)}")

                elif not temp_data_for_stats.empty:
                    counts = temp_data_for_stats.value_counts()
                    top_categories = [f"'{str(i)}' ({counts[i]})" for i in counts.head(3).index]
                    summary_text = (
                        f"\n\n**Summary for '{get_original_name(primary_plot_col)}':** {counts.size} unique values. "
                        f"Top: {', '.join(top_categories)}.")

                response_text += summary_text

        return chart, response_text

    except Exception as e:
        return None, f"Error generating visualization: {e}"


def create_insights_chart(data, labels=None, values=None, chart_type='pie', title=None):
    """
    Create a simple chart for analytics insights.

    Args:
        data: DataFrame containing the data
        labels: Column to use for labels/categories
        values: Column to use for values (for bar/line charts)
        chart_type: Type of chart ('pie', 'bar', etc.)
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        if chart_type == 'pie' and labels:
            counts = data[labels].value_counts().reset_index()
            counts.columns = [labels, 'count']
            fig = px.pie(counts, names=labels, values='count', title=title)
            return fig

        elif chart_type == 'bar' and labels:
            if values:
                # Bar chart with specific values
                fig = px.bar(data, x=labels, y=values, title=title)
            else:
                # Bar chart showing counts
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