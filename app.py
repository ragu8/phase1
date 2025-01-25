import pandas as pd
import plotly.express as px
import streamlit as st

# Load CSV File
def load_csv(file):
    return pd.read_csv(file)

# Get Best Model
def get_best_model(df, metric="Accuracy"):
    return df.loc[df[metric].idxmax()]

# Compare Models
def compare_models(df, metric="Accuracy"):
    sorted_df = df.sort_values(by=metric, ascending=True)
    lowest_2 = sorted_df.head(2)
    best_3 = sorted_df.tail(3)
    return lowest_2, best_3

# Plot Model Comparison using Plotly for Interactivity
def plot_comparison_interactive(lowest_2, best_3, metric):
    comparison_df = pd.concat([lowest_2, best_3])
    comparison_df['Model Label'] = comparison_df['Feature Model'] + " - " + comparison_df['Model']

    fig = px.bar(
        comparison_df,
        x=metric,
        y="Model Label",
        color="Model Label",
        title=f"Model Comparison Based on {metric}",
        labels={metric: metric, 'Model Label': 'Models'},
        color_discrete_sequence=["#ff9999"] * 2 + ["#99ff99"] * 3
    )
    st.plotly_chart(fig)

# Plot Model Comparison for Two Models (Interactive)
def plot_two_model_comparison_interactive(df, metric, feature_model_1, model_1, feature_model_2, model_2):
    # Filter the two selected models and feature models
    df_filtered = df[(df['Feature Model'].isin([feature_model_1, feature_model_2])) & 
                     (df['Model'].isin([model_1, model_2]))]
    
    # Interactive Bar Plot using Plotly
    fig = px.bar(
        df_filtered,
        x='Model',
        y=metric,
        color='Feature Model',
        title=f"Comparison Between {model_1} ({feature_model_1}) and {model_2} ({feature_model_2}) Based on {metric}",
        labels={metric: metric, 'Model': 'Model'},
        color_discrete_sequence=["#ff9999", "#99ff99"]
    )
    st.plotly_chart(fig)

# Streamlit App Layout
def main():
    st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
    st.title("📊 Model Comparison Dashboard")
    st.write("This dashboard helps analyze and compare model performances interactively.")

    # Sidebar for Upload and Metric Selection
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your model results CSV", type=["csv"])

    # Metric Selection Dropdown
    metric_options = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric = st.sidebar.selectbox("Select Metric for Comparison", metric_options)

    # Check for Uploaded File
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.sidebar.success("File Uploaded Successfully!")

        # Display Uploaded Data
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df)

        # Display Best Model
        st.subheader(f"🏆 Best Model Based on {metric}")
        best_model = get_best_model(df, metric)
        if best_model is not None:
            st.json({
                "Feature Model": best_model['Feature Model'],
                "Model": best_model['Model'],
                "Parameters": best_model['Parameters'],
                metric: best_model[metric],
                "Precision": best_model['Precision'],
                "Recall": best_model['Recall'],
                "F1-Score": best_model['F1-Score'],
                "Time Taken (s)": best_model['Time Taken (s)'],
                "Class Names": best_model['Class Names']
            })

        # Compare Models
        st.subheader("📈 Model Comparison")
        lowest_2, best_3 = compare_models(df, metric)
        st.write("**Comparison Table:**")
        st.dataframe(pd.concat([lowest_2, best_3])[["Feature Model", "Model", "Parameters", metric]])

        # Plot Comparison (Bar plot)
        st.subheader("📊 Visual Comparison of Models (Interactive Bar Plot)")
        plot_comparison_interactive(lowest_2, best_3, metric)

        # Add a section to choose two specific models and feature models for comparison
        st.subheader("🔍 Compare Two Models")
        feature_model_options = df['Feature Model'].unique()
        model_options = df['Model'].unique()
        
        feature_model_1 = st.selectbox("Select Feature Model for First Model", feature_model_options)
        model_1 = st.selectbox("Select First Model", model_options)

        feature_model_2 = st.selectbox("Select Feature Model for Second Model", feature_model_options)
        model_2 = st.selectbox("Select Second Model", model_options)

        # Show the comparison with details for both models
        if model_1 != model_2:
            st.subheader(f"📊 Comparison Between {model_1} ({feature_model_1}) and {model_2} ({feature_model_2})")
            
            # Filter and display the details of both models selected
            model_1_details = df[(df['Feature Model'] == feature_model_1) & (df['Model'] == model_1)].iloc[0]
            model_2_details = df[(df['Feature Model'] == feature_model_2) & (df['Model'] == model_2)].iloc[0]
            
            st.write(f"### {model_1} ({feature_model_1})")
            st.write(f"- **Model**: {model_1}")
            st.write(f"- **Feature Model**: {feature_model_1}")
            st.write(f"- **Parameters**: {model_1_details['Parameters']}")
            st.write(f"- **{metric}**: {model_1_details[metric]}")
            st.write(f"- **Precision**: {model_1_details['Precision']}")
            st.write(f"- **Recall**: {model_1_details['Recall']}")
            st.write(f"- **F1-Score**: {model_1_details['F1-Score']}")
            st.write(f"- **Time Taken (s)**: {model_1_details['Time Taken (s)']}")

            st.write(f"### {model_2} ({feature_model_2})")
            st.write(f"- **Model**: {model_2}")
            st.write(f"- **Feature Model**: {feature_model_2}")
            st.write(f"- **Parameters**: {model_2_details['Parameters']}")
            st.write(f"- **{metric}**: {model_2_details[metric]}")
            st.write(f"- **Precision**: {model_2_details['Precision']}")
            st.write(f"- **Recall**: {model_2_details['Recall']}")
            st.write(f"- **F1-Score**: {model_2_details['F1-Score']}")
            st.write(f"- **Time Taken (s)**: {model_2_details['Time Taken (s)']}")

            # Plot the comparison for the selected models
            plot_two_model_comparison_interactive(df, metric, feature_model_1, model_1, feature_model_2, model_2)
        else:
            st.warning("Please select two different models to compare.")

        # Add Summary
        st.subheader("📝 Inference Summary")
        st.write(
            f"- **Best Model**: {best_model['Feature Model']} - {best_model['Model']} with {metric}: **{best_model[metric]}**\n"
            "- **Lowest Models** are shown in red, while the best-performing models are in green.\n"
            "- Use the visual chart to understand model performance."
        )
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
