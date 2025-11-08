import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from logistic_from_scratch import (
    LogisticRegressionFromScratch,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, standardize_features
)


def validate_and_clean_data(df, target, task_type='classification'):
    """
    Validate and clean uploaded data for classification.
    Returns cleaned dataframe, target, and any warnings/errors.
    """
    errors = []
    warnings = []
    
    if df is None or df.empty:
        errors.append("Dataset is empty or could not be loaded.")
        return None, None, errors, warnings
    
    if target is None or len(target) == 0:
        errors.append("Target column is empty or invalid.")
        return None, None, errors, warnings
    
    if len(df) != len(target):
        errors.append(f"Feature count ({len(df)}) doesn't match target count ({len(target)}).")
        return None, None, errors, warnings
    
    df_cleaned = df.copy()
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if len(non_numeric_cols) > 0:
        warnings.append(f"Found {len(non_numeric_cols)} non-numeric column(s): {', '.join(non_numeric_cols)}. These will be dropped.")
        df_cleaned = df_cleaned[numeric_cols]
    
    if df_cleaned.empty:
        errors.append("No numeric features remaining after cleaning. Please ensure your dataset contains numeric columns.")
        return None, None, errors, warnings
    
    missing_values = df_cleaned.isnull().sum().sum()
    if missing_values > 0:
        warnings.append(f"Found {missing_values} missing values. These will be filled with column means.")
        df_cleaned = df_cleaned.fillna(df_cleaned.mean())
    
    try:
        target = np.array(target)
        unique_classes = np.unique(target)
        
        if task_type == 'classification':
            if len(unique_classes) < 2:
                errors.append("Classification requires at least 2 classes in target.")
                return None, None, errors, warnings
            
            if len(unique_classes) > 2:
                errors.append("This implementation currently supports binary classification only. Please ensure your target has exactly 2 classes.")
                return None, None, errors, warnings
            
            target = target.astype(int)
        else:
            target = np.array(target, dtype=float)
            if np.any(np.isnan(target)):
                warnings.append("Found missing values in target. These rows will be removed.")
                valid_mask = ~np.isnan(target)
                target = target[valid_mask]
                df_cleaned = df_cleaned.iloc[valid_mask]
    except (ValueError, TypeError) as e:
        errors.append(f"Target column contains invalid values: {str(e)}")
        return None, None, errors, warnings
    
    if len(df_cleaned) < 10:
        errors.append(f"Insufficient data: need at least 10 samples, got {len(df_cleaned)}.")
        return None, None, errors, warnings
    
    if df_cleaned.shape[1] < 1:
        errors.append("Need at least 1 numeric feature for classification.")
        return None, None, errors, warnings
    
    return df_cleaned, target, errors, warnings


def load_sample_datasets():
    """Load built-in classification datasets for demonstration."""
    datasets = {}
    
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        datasets['Iris (Binary)'] = {
            'data': pd.DataFrame(iris.data[iris.target != 2], columns=iris.feature_names),
            'target': iris.target[iris.target != 2],
            'target_names': iris.target_names[:2],
            'description': 'Iris dataset (setosa vs versicolor) - 2 classes, 4 features'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        datasets['Breast Cancer'] = {
            'data': pd.DataFrame(cancer.data, columns=cancer.feature_names),
            'target': cancer.target,
            'target_names': cancer.target_names,
            'description': 'Breast cancer diagnostic dataset - binary classification'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_wine
        wine = load_wine()
        datasets['Wine (Binary)'] = {
            'data': pd.DataFrame(wine.data[wine.target != 2], columns=wine.feature_names),
            'target': wine.target[wine.target != 2],
            'target_names': wine.target_names[:2],
            'description': 'Wine dataset (class 0 vs 1) - binary classification'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                                   n_redundant=1, n_clusters_per_class=1, 
                                   random_state=42)
        feature_names = [f'Feature_{i+1}' for i in range(4)]
        datasets['Synthetic Data'] = {
            'data': pd.DataFrame(X, columns=feature_names),
            'target': y,
            'target_names': np.array(['Class 0', 'Class 1']),
            'description': 'Synthetic binary classification dataset with 4 features'
        }
    except ImportError:
        pass
    
    return datasets


def create_confusion_matrix_plot(cm, class_names):
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    
    return fig


def create_probability_distribution_plot(y_true, y_proba):
    """Create probability distribution plot."""
    df_plot = pd.DataFrame({
        'Probability': y_proba,
        'True Class': ['Class 1' if y == 1 else 'Class 0' for y in y_true]
    })
    
    fig = px.histogram(
        df_plot, x='Probability', color='True Class',
        nbins=30, barmode='overlay',
        title='Predicted Probability Distribution',
        labels={'Probability': 'Predicted Probability for Class 1'}
    )
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Decision Threshold")
    
    fig.update_layout(width=700, height=500)
    return fig


def create_roc_curve_plot(y_true, y_proba):
    """Create ROC curve plot."""
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    auc = np.trapz(tpr_list, fpr_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr_list, y=tpr_list,
        mode='lines',
        name=f'ROC Curve (AUC = {abs(auc):.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    
    return fig


def create_decision_boundary_plot(X, y, model, feature_names):
    """Create decision boundary plot for 2D data."""
    if X.shape[1] != 2:
        return None
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
        showscale=False,
        opacity=0.5
    ))
    
    for class_val in [0, 1]:
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Decision Boundary',
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        width=700,
        height=500
    )
    
    return fig


def create_cost_history_plot(cost_history, title):
    """Create line plot of cost history for gradient descent."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(cost_history))),
        y=cost_history,
        mode='lines',
        name='Cost',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title='Binary Cross-Entropy Loss',
        width=700,
        height=500
    )
    
    return fig


def create_feature_importance_plot(coefficients, feature_names, title):
    """Create bar plot of feature coefficients."""
    df_plot = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    df_plot = df_plot.sort_values('Coefficient', key=abs, ascending=True)
    
    fig = px.bar(
        df_plot, x='Coefficient', y='Feature',
        orientation='h',
        title=title,
        labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'}
    )
    
    fig.update_layout(width=700, height=max(400, len(feature_names) * 30))
    return fig


def main():
    st.set_page_config(
        page_title="Logistic Regression from Scratch",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Logistic Regression from Scratch")
    
    st.sidebar.header("Data Selection")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Built-in Datasets", "Upload CSV"]
    )
    
    df = None
    target = None
    target_names = None
    dataset_name = ""
    
    if data_source == "Built-in Datasets":
        datasets = load_sample_datasets()
        
        if not datasets:
            st.error("No built-in datasets available. Please install scikit-learn.")
            return
        
        dataset_choice = st.sidebar.selectbox(
            "Select dataset:",
            list(datasets.keys())
        )
        
        if dataset_choice:
            dataset = datasets[dataset_choice]
            df = dataset['data']
            target = dataset['target']
            target_names = dataset['target_names']
            dataset_name = dataset_choice
            
            st.sidebar.info(f"**{dataset_choice}**\n\n{dataset['description']}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
                
                if len(df.columns) < 2:
                    st.error("Dataset must have at least 2 columns (features + target).")
                    df = None
                else:
                    target_col = st.sidebar.selectbox(
                        "Select target column:",
                        list(df.columns)
                    )
                    
                    if target_col:
                        target = df[target_col].values
                        df = df.drop(columns=[target_col])
                        
                        df, target, errors, warnings = validate_and_clean_data(df, target, 'classification')
                        
                        if errors:
                            for error in errors:
                                st.error(error)
                            df = None
                            target = None
                            target_names = None
                        else:
                            if warnings:
                                for warning in warnings:
                                    st.warning(warning)
                            unique_classes = np.unique(target)
                            target_names = unique_classes if len(unique_classes) == 2 else None
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                df = None
                target = None
                target_names = None
    
    if df is not None and target is not None:
        st.header(f"Dataset: {dataset_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            n_classes = len(np.unique(target))
            st.metric("Classes", n_classes)
        
        if n_classes != 2:
            st.warning("This implementation currently supports binary classification only. Please select a binary dataset.")
            return
        
        with st.expander("Dataset Preview"):
            preview_df = df.copy()
            preview_df['Target'] = target
            st.dataframe(preview_df.head(10))
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_selection = st.multiselect(
                "Select features (leave empty for all):",
                list(df.columns),
                default=[]
            )
        
        if len(feature_selection) == 0:
            feature_selection = list(df.columns)
        
        try:
            X = df[feature_selection].select_dtypes(include=[np.number]).values
            if X.shape[1] == 0:
                st.error("Selected features contain no numeric data. Please select numeric features only.")
                st.stop()
            
            y = np.array(target, dtype=int)
            
            if X.shape[0] != len(y):
                st.error("Feature and target dimensions don't match.")
                st.stop()
            
            from sklearn.model_selection import train_test_split
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            if len(X_train) < 2:
                st.error("Training set too small. Please reduce test size or use more data.")
                st.stop()
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.stop()
        
        st.markdown("#### Gradient Descent Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.4f"
            )
        
        with col2:
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=10,
                max_value=10000,
                value=1000,
                step=100
            )
        
        with col3:
            batch_type = st.selectbox(
                "Gradient Descent Type:",
                ["Batch", "Mini-Batch", "Stochastic"]
            )
        
        if batch_type == "Mini-Batch":
            batch_size = st.slider(
                "Batch Size",
                min_value=2,
                max_value=min(len(X_train), 100),
                value=32
            )
        elif batch_type == "Stochastic":
            batch_size = 1
        else:
            batch_size = None
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    X_train_scaled, mean, std = standardize_features(X_train)
                    X_test_scaled = (X_test - mean) / std
                    
                    model = LogisticRegressionFromScratch(
                        learning_rate=learning_rate,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        fit_intercept=True
                    )
                    model.fit(X_train_scaled, y_train, verbose=False)
                    
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    y_train_proba = model.predict_proba(X_train_scaled)
                    y_test_proba = model.predict_proba(X_test_scaled)
                    
                    st.session_state['model'] = model
                    st.session_state['X_train'] = X_train_scaled
                    st.session_state['X_test'] = X_test_scaled
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['y_train_pred'] = y_train_pred
                    st.session_state['y_test_pred'] = y_test_pred
                    st.session_state['y_train_proba'] = y_train_proba
                    st.session_state['y_test_proba'] = y_test_proba
                    st.session_state['feature_names'] = feature_selection
                    st.session_state['target_names'] = target_names
                except ValueError as e:
                    st.error(f"Value error during training: {str(e)}")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            y_train_pred = st.session_state['y_train_pred']
            y_test_pred = st.session_state['y_test_pred']
            y_train_proba = st.session_state['y_train_proba']
            y_test_proba = st.session_state['y_test_proba']
            feature_names = st.session_state['feature_names']
            target_names = st.session_state['target_names']
            
            st.header("Model Results")
            
            train_acc = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Accuracy", f"{train_acc:.4f}")
                    st.metric("Precision", f"{train_precision:.4f}")
                with met_col2:
                    st.metric("Recall", f"{train_recall:.4f}")
                    st.metric("F1 Score", f"{train_f1:.4f}")
            
            with col2:
                st.subheader("Test Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Accuracy", f"{test_acc:.4f}")
                    st.metric("Precision", f"{test_precision:.4f}")
                with met_col2:
                    st.metric("Recall", f"{test_recall:.4f}")
                    st.metric("F1 Score", f"{test_f1:.4f}")
            
            st.markdown("---")
            
            with st.expander("Model Coefficients"):
                st.write(f"**Intercept:** {model.intercept_:.4f}")
                st.write("**Coefficients:**")
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coefficients_
                })
                st.dataframe(coef_df)
            
            st.subheader("Visualizations")
            
            tab_names = ["Confusion Matrix", "Probability Distribution", "ROC Curve", "Feature Importance", "Cost History"]
            
            if len(feature_names) == 2:
                tab_names.append("Decision Boundary")
            
            viz_tabs = st.tabs(tab_names)
            
            with viz_tabs[0]:
                cm = confusion_matrix(y_test, y_test_pred)
                fig_cm = create_confusion_matrix_plot(cm, target_names)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.info("""
                **Confusion Matrix:**
                - Shows the distribution of actual vs predicted classifications
                - Diagonal elements show correct predictions
                - Off-diagonal elements show misclassifications
                """)
            
            with viz_tabs[1]:
                fig_prob = create_probability_distribution_plot(y_test, y_test_proba)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                st.info("""
                **Probability Distribution:**
                - Shows how confident the model is in its predictions
                - Good separation between classes indicates strong performance
                - Overlap near 0.5 threshold indicates uncertain predictions
                """)
            
            with viz_tabs[2]:
                fig_roc = create_roc_curve_plot(y_test, y_test_proba)
                st.plotly_chart(fig_roc, use_container_width=True)
                
                st.info("""
                **ROC Curve:**
                - Shows trade-off between true positive rate and false positive rate
                - Area Under Curve (AUC) summarizes overall performance
                - AUC = 0.5 is random, AUC = 1.0 is perfect
                - Closer to top-left corner is better
                """)
            
            with viz_tabs[3]:
                fig_importance = create_feature_importance_plot(
                    model.coefficients_,
                    feature_names,
                    "Feature Coefficients"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.info("""
                **Feature Coefficients:**
                - Shows the impact of each feature on the classification decision
                - Positive coefficients increase probability of Class 1
                - Negative coefficients decrease probability of Class 1
                - Larger absolute values indicate stronger influence
                """)
            
            with viz_tabs[4]:
                fig_cost = create_cost_history_plot(
                    model.cost_history_,
                    f"Cost History - {model.get_batch_type()}"
                )
                st.plotly_chart(fig_cost, use_container_width=True)
                
                st.info(f"""
                **Cost History ({model.get_batch_type()}):**
                - Shows how the model's loss decreased during training
                - Decreasing curve indicates successful learning
                - Flattening indicates convergence
                - Final cost: {model.cost_history_[-1]:.4f}
                """)
            
            if len(feature_names) == 2:
                with viz_tabs[5]:
                    fig_boundary = create_decision_boundary_plot(
                        X_test, y_test, model, feature_names
                    )
                    if fig_boundary:
                        st.plotly_chart(fig_boundary, use_container_width=True)
                        
                        st.info("""
                        **Decision Boundary:**
                        - Shows how the model separates the two classes
                        - Colors represent predicted regions
                        - Points show actual data samples
                        - Clear separation indicates good classification
                        """)
    
    else:
        st.info("Please select a data source from the sidebar to get started!")


if __name__ == "__main__":
    main()

