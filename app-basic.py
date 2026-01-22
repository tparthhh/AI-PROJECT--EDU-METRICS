import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="EduMetrics: Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Title
st.markdown('<p class="main-header">📚 EduMetrics: Predict & Progress</p>', unsafe_allow_html=True)
st.markdown("### AI-Powered Student Performance Analysis System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("Navigation")
    page = st.radio("", ["🏠 Home", "📤 Upload & Predict", "📊 Analytics Dashboard", "🎯 Individual Prediction", "ℹ️ Model Info", "👥 About"])
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    if st.session_state.df is not None:
        st.metric("Students Loaded", len(st.session_state.df))
        if st.session_state.predictions_made:
            st.metric("Predictions Made", "✓", delta="Ready")
    else:
        st.info("No data loaded yet")

# HOME PAGE
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to EduMetrics! 👋")
        st.markdown("""
        ### Your AI-Powered Academic Performance Assistant
        
        EduMetrics helps educators, students, and parents make data-driven decisions about academic performance.
        
        #### 🎯 What Can You Do?
        
        1. **Upload Student Data** - Import CSV files with student performance metrics
        2. **Train AI Models** - Use Random Forest or Linear Regression algorithms
        3. **Generate Predictions** - Forecast student performance and identify at-risk students
        4. **Visualize Insights** - Interactive charts and comprehensive analytics
        5. **Get Recommendations** - Actionable intervention strategies
        6. **Individual Predictions** - Predict performance for new students
        
        #### 🚀 Getting Started
        
        1. Navigate to **"📤 Upload & Predict"** from the sidebar
        2. Download the sample template or upload your own data
        3. Select features and train the model
        4. View predictions and analytics
        
        #### 💡 Key Features
        
        - **Dual Algorithm Support**: Random Forest & Linear Regression
        - **Real-time Predictions**: Instant results after training
        - **Risk Classification**: Automatic categorization into risk levels
        - **Feature Importance**: Understand what drives performance
        - **Cross-Validation**: Robust model evaluation
        - **Export Results**: Download predictions as CSV
        """)
    
    with col2:
        st.info("### 📋 Quick Guide")
        st.markdown("""
        **Required Data Columns:**
        - Student ID (optional)
        - Attendance (%)
        - Study Hours
        - Previous Score
        - Sleep Hours
        - Stress Level
        - Final Score (target)
        
        **Supported Features:**
        - Batch predictions
        - Individual student scoring
        - Comparative analysis
        - Trend visualization
        """)
        
        st.success("### ✨ New Features!")
        st.markdown("""
        - Model comparison
        - Cross-validation scores
        - Enhanced visualizations
        - Individual student predictor
        - Detailed error metrics
        """)

# UPLOAD & PREDICT PAGE
elif page == "📤 Upload & Predict":
    st.header("Student Performance Prediction")
    
    # File upload section
    st.subheader("Step 1: Upload Student Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with student information",
            type=['csv'],
            help="Upload a CSV file containing student performance data"
        )
    
    with col2:
        st.info("💡 **Tip**\n\nDownload the template below if you don't have data")
    
    # Sample data template
    with st.expander("📋 View & Download Sample Template"):
        sample_data = pd.DataFrame({
            'Student_ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'Attendance': [85, 92, 78, 95, 70],
            'Study_Hours': [4, 6, 3, 7, 2],
            'Previous_Score': [75, 88, 65, 92, 60],
            'Sleep_Hours': [7, 8, 6, 7, 5],
            'Stress_Level': [3, 2, 4, 2, 5],
            'Final_Score': [78, 90, 67, 94, 62]
        })
        st.dataframe(sample_data, use_container_width=True)
        
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Sample Template",
            csv_buffer.getvalue(),
            "student_data_template.csv",
            "text/csv",
            key='download-csv'
        )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.success(f"✅ Successfully loaded data for {len(df)} students")
            
            # Data validation
            if len(df) < 5:
                st.warning("⚠️ Dataset is very small. Consider adding more samples for better predictions.")
            
            # Display data preview
            st.subheader("Step 2: Data Preview & Quality Check")
            
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Statistics", "🔍 Data Quality"])
            
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(df)} rows")
            
            with tab2:
                st.write("**Descriptive Statistics**")
                st.dataframe(df.describe(), use_container_width=True)
            
            with tab3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    missing_values = df.isnull().sum().sum()
                    st.metric("Missing Values", missing_values, 
                             delta="Good" if missing_values == 0 else "Needs attention",
                             delta_color="normal" if missing_values == 0 else "inverse")
                with col2:
                    duplicate_rows = df.duplicated().sum()
                    st.metric("Duplicate Rows", duplicate_rows,
                             delta="Good" if duplicate_rows == 0 else "Remove duplicates",
                             delta_color="normal" if duplicate_rows == 0 else "inverse")
                with col3:
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.metric("Numeric Columns", numeric_cols)
                
                if missing_values > 0:
                    st.warning("Missing values detected. They will be filled with column means during training.")
                    st.write("**Missing Values by Column:**")
                    missing_df = df.isnull().sum()
                    missing_df = missing_df[missing_df > 0].sort_values(ascending=False)
                    st.write(missing_df)
            
            # Feature selection
            st.subheader("Step 3: Configure Model")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("❌ Not enough numeric columns for prediction. Please upload data with at least 2 numeric columns.")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Smart default feature selection
                    default_features = [col for col in ['Attendance', 'Study_Hours', 'Previous_Score', 'Sleep_Hours', 'Stress_Level'] 
                                      if col in numeric_columns]
                    if not default_features:
                        default_features = numeric_columns[:min(5, len(numeric_columns)-1)]
                    
                    features = st.multiselect(
                        "Select Input Features",
                        numeric_columns,
                        default=default_features,
                        help="Choose the features that will be used to make predictions"
                    )
                
                with col2:
                    # Smart target selection
                    available_targets = [col for col in numeric_columns if col not in features]
                    default_target = next((col for col in ['Final_Score', 'Score', 'Grade'] if col in available_targets), 
                                        available_targets[0] if available_targets else None)
                    
                    target = st.selectbox(
                        "Select Target Variable",
                        available_targets if features else numeric_columns,
                        index=available_targets.index(default_target) if default_target and default_target in available_targets else 0,
                        help="This is what the model will predict"
                    )
                
                with col3:
                    algorithm = st.selectbox(
                        "Select Algorithm",
                        ["Random Forest", "Linear Regression"],
                        help="Random Forest: Better for complex patterns\nLinear Regression: Better for simple linear relationships"
                    )
                
                # Advanced settings
                with st.expander("⚙️ Advanced Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
                        if algorithm == "Random Forest":
                            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
                    with col2:
                        random_state = st.number_input("Random Seed", 1, 100, 42)
                        use_cross_validation = st.checkbox("Use Cross-Validation", value=True,
                                                          help="More robust but slower")
                
                # Train button
                if st.button("🚀 Train Model & Generate Predictions", type="primary", use_container_width=True):
                    if len(features) == 0:
                        st.error("❌ Please select at least one feature!")
                    elif target is None:
                        st.error("❌ Please select a target variable!")
                    else:
                        with st.spinner(f"Training {algorithm} model... Please wait."):
                            try:
                                # Prepare data
                                X = df[features].copy()
                                y = df[target].copy()
                                
                                # Handle missing values
                                X = X.fillna(X.mean())
                                y = y.fillna(y.mean())
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size/100, random_state=int(random_state)
                                )
                                
                                # Train model
                                if algorithm == "Random Forest":
                                    model = RandomForestRegressor(
                                        n_estimators=n_estimators,
                                        random_state=int(random_state),
                                        n_jobs=-1,
                                        max_depth=10
                                    )
                                else:
                                    model = LinearRegression()
                                
                                model.fit(X_train, y_train)
                                
                                # Make predictions
                                y_pred_train = model.predict(X_train)
                                y_pred_test = model.predict(X_test)
                                predictions = model.predict(X)
                                
                                # Calculate metrics
                                train_score = r2_score(y_train, y_pred_train)
                                test_score = r2_score(y_test, y_pred_test)
                                mae = mean_absolute_error(y_test, y_pred_test)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                                
                                # Cross-validation
                                if use_cross_validation:
                                    cv_scores = cross_val_score(model, X, y, cv=5, 
                                                               scoring='r2')
                                    cv_mean = cv_scores.mean()
                                    cv_std = cv_scores.std()
                                
                                # Save to session state
                                st.session_state.model = model
                                st.session_state.features = features
                                st.session_state.target = target
                                st.session_state.model_trained = True
                                st.session_state.predictions_made = True
                                
                                # Add predictions to dataframe
                                df['Predicted_Score'] = predictions
                                df['Prediction_Error'] = abs(y - predictions)
                                
                                # Risk classification with more granular levels
                                df['Risk_Level'] = pd.cut(
                                    predictions,
                                    bins=[0, 40, 60, 75, 100],
                                    labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent']
                                )
                                
                                st.session_state.df = df
                                
                                st.success(f"✅ Model trained successfully!")
                                
                                # Display metrics
                                st.subheader("Step 4: Model Performance")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Training R² Score", f"{train_score:.3f}")
                                with col2:
                                    st.metric("Testing R² Score", f"{test_score:.3f}",
                                             delta=f"{(test_score - train_score):.3f}")
                                with col3:
                                    st.metric("Mean Abs Error", f"{mae:.2f}")
                                with col4:
                                    st.metric("Root MSE", f"{rmse:.2f}")
                                
                                if use_cross_validation:
                                    st.info(f"📊 Cross-Validation Score: {cv_mean:.3f} (±{cv_std:.3f})")
                                
                                # Model interpretation
                                if test_score > 0.8:
                                    st.success("🎉 Excellent model performance! Predictions are highly reliable.")
                                elif test_score > 0.6:
                                    st.info("👍 Good model performance. Predictions are reasonably reliable.")
                                else:
                                    st.warning("⚠️ Model performance could be improved. Consider adding more data or features.")
                                
                                # Display results
                                st.subheader("Step 5: Prediction Results")
                                
                                # Risk metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Students", len(df))
                                with col2:
                                    high_risk = len(df[df['Risk_Level'] == 'High Risk'])
                                    st.metric("High Risk", high_risk, 
                                             delta=f"{(high_risk/len(df)*100):.1f}%")
                                with col3:
                                    medium_risk = len(df[df['Risk_Level'] == 'Medium Risk'])
                                    st.metric("Medium Risk", medium_risk,
                                             delta=f"{(medium_risk/len(df)*100):.1f}%")
                                with col4:
                                    low_risk = len(df[df['Risk_Level'] == 'Low Risk'])
                                    excellent = len(df[df['Risk_Level'] == 'Excellent'])
                                    st.metric("Low Risk + Excellent", low_risk + excellent,
                                             delta=f"{((low_risk+excellent)/len(df)*100):.1f}%")
                                
                            except Exception as e:
                                st.error(f"❌ Error during training: {str(e)}")
                                st.info("Please check your data format and try again.")
                                import traceback
                                with st.expander("Show error details"):
                                    st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
            with st.expander("Show error details"):
                st.code(str(e))

# ANALYTICS DASHBOARD
elif page == "📊 Analytics Dashboard":
    st.header("Analytics Dashboard")
    
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded. Please upload data first in the 'Upload & Predict' page.")
    elif not st.session_state.predictions_made:
        st.warning("⚠️ No predictions made yet. Please train a model first.")
    else:
        df = st.session_state.df
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Risk Distribution", 
            "📈 Predictions Table", 
            "🎯 Feature Importance", 
            "📉 Performance Analysis",
            "💡 Recommendations"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = df['Risk_Level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Student Risk Level Distribution",
                    color=risk_counts.index,
                    color_discrete_map={
                        'High Risk': '#ef4444',
                        'Medium Risk': '#f59e0b',
                        'Low Risk': '#10b981',
                        'Excellent': '#3b82f6'
                    },
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score distribution
                fig2 = px.histogram(
                    df,
                    x='Predicted_Score',
                    nbins=20,
                    title="Predicted Score Distribution",
                    labels={'Predicted_Score': 'Predicted Score'},
                    color_discrete_sequence=['#6366f1']
                )
                fig2.add_vline(x=df['Predicted_Score'].mean(), 
                              line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {df['Predicted_Score'].mean():.1f}")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Box plot
            fig3 = px.box(
                df,
                x='Risk_Level',
                y='Predicted_Score',
                title="Score Distribution by Risk Level",
                color='Risk_Level',
                color_discrete_map={
                    'High Risk': '#ef4444',
                    'Medium Risk': '#f59e0b',
                    'Low Risk': '#10b981',
                    'Excellent': '#3b82f6'
                }
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            # Display predictions with color coding
            st.subheader("Detailed Predictions Table")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    df['Risk_Level'].unique(),
                    default=df['Risk_Level'].unique()
                )
            with col2:
                score_range = st.slider(
                    "Filter by Predicted Score Range",
                    float(df['Predicted_Score'].min()),
                    float(df['Predicted_Score'].max()),
                    (float(df['Predicted_Score'].min()), float(df['Predicted_Score'].max()))
                )
            
            # Filter dataframe
            filtered_df = df[
                (df['Risk_Level'].isin(risk_filter)) &
                (df['Predicted_Score'] >= score_range[0]) &
                (df['Predicted_Score'] <= score_range[1])
            ]
            
            st.info(f"Showing {len(filtered_df)} of {len(df)} students")
            
            # Prepare display dataframe
            display_columns = []
            if 'Student_ID' in df.columns:
                display_columns.append('Student_ID')
            display_columns.extend(st.session_state.features)
            if st.session_state.target in df.columns:
                display_columns.append(st.session_state.target)
            display_columns.extend(['Predicted_Score', 'Prediction_Error', 'Risk_Level'])
            
            display_df = filtered_df[display_columns].copy()
            display_df['Predicted_Score'] = display_df['Predicted_Score'].round(2)
            display_df['Prediction_Error'] = display_df['Prediction_Error'].round(2)
            
            # Color coding function
            def highlight_risk(row):
                if row['Risk_Level'] == 'High Risk':
                    return ['background-color: #fee2e2'] * len(row)
                elif row['Risk_Level'] == 'Medium Risk':
                    return ['background-color: #fef3c7'] * len(row)
                elif row['Risk_Level'] == 'Low Risk':
                    return ['background-color: #d1fae5'] * len(row)
                else:
                    return ['background-color: #dbeafe'] * len(row)
            
            st.dataframe(
                display_df.style.apply(highlight_risk, axis=1),
                use_container_width=True,
                height=400
            )
            
            # Download results
            col1, col2 = st.columns(2)
            with col1:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Full Results",
                    csv_buffer.getvalue(),
                    f"student_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                csv_buffer2 = io.StringIO()
                filtered_df.to_csv(csv_buffer2, index=False)
                st.download_button(
                    "📥 Download Filtered Results",
                    csv_buffer2.getvalue(),
                    f"filtered_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with tab3:
            # Feature importance
            if hasattr(st.session_state.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Importance': st.session_state.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Predictions",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Higher importance means the feature has more influence on predictions")
                
                # Feature importance table
                st.subheader("Feature Importance Rankings")
                importance_df['Importance_Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100).round(2)
                st.dataframe(importance_df, use_container_width=True)
                
            elif hasattr(st.session_state.model, 'coef_'):
                # For Linear Regression
                coef_df = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Coefficient': st.session_state.model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                fig = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title="Feature Coefficients (Linear Regression)",
                    color='Coefficient',
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Positive coefficients increase the score, negative decrease it")
                st.dataframe(coef_df, use_container_width=True)
        
        with tab4:
            st.subheader("Performance Analysis")
            
            # Actual vs Predicted scatter plot
            if st.session_state.target in df.columns:
                fig = px.scatter(
                    df,
                    x=st.session_state.target,
                    y='Predicted_Score',
                    color='Risk_Level',
                    title=f"Actual vs Predicted {st.session_state.target}",
                    labels={
                        st.session_state.target: f'Actual {st.session_state.target}',
                        'Predicted_Score': 'Predicted Score'
                    },
                    color_discrete_map={
                        'High Risk': '#ef4444',
                        'Medium Risk': '#f59e0b',
                        'Low Risk': '#10b981',
                        'Excellent': '#3b82f6'
                    },
                    trendline="ols"
                )
                
                # Add perfect prediction line
                min_val = min(df[st.session_state.target].min(), df['Predicted_Score'].min())
                max_val = max(df[st.session_state.target].max(), df['Predicted_Score'].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot
                df['Residual'] = df[st.session_state.target] - df['Predicted_Score']
                fig2 = px.scatter(
                    df,
                    x='Predicted_Score',
                    y='Residual',
                    color='Risk_Level',
                    title="Residual Plot (Prediction Errors)",
                    labels={'Residual': 'Error (Actual - Predicted)'},
                    color_discrete_map={
                        'High Risk': '#ef4444',
                        'Medium Risk': '#f59e0b',
                        'Low Risk': '#10b981',
                        'Excellent': '#3b82f6'
                    }
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Error distribution
                fig3 = px.histogram(
                    df,
                    x='Prediction_Error',
                    nbins=20,
                    title="Prediction Error Distribution",
                    labels={'Prediction_Error': 'Absolute Prediction Error'},
                    color_discrete_sequence=['#8b5cf6']
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab5:
            st.markdown("### 🎯 Personalized Recommendations")
            
            high_risk = df[df['Risk_Level'] == 'High Risk']
            if len(high_risk) > 0:
                st.error(f"⚠️ {len(high_risk)} students identified as High Risk")
                st.markdown("**Immediate Actions Required:**")
                st.markdown("""
                - 🚨 Schedule urgent one-on-one counseling sessions
                - 👥 Assign dedicated peer tutors or mentors
                - 📝 Create personalized, intensive study plans
                - 📊 Increase monitoring to weekly check-ins
                - 📞 Contact parents/guardians immediately
                - 🎯 Set short-term achievable goals (weekly milestones)
                """)
                
                with st.expander("📋 View High Risk Students Details"):
                    st.dataframe(high_risk, use_container_width=True)
                    
                    # Top factors for high risk students
                    if hasattr(st.session_state.model, 'feature_importances_'):
                        st.write("**Key Areas to Focus On (based on feature importance):**")
                        importance_df = pd.DataFrame({
                            'Feature': st.session_state.features,
                            'Importance': st.session_state.model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        for idx in range(min(3, len(importance_df))):
                            st.write(f"• {importance_df.iloc[idx]['Feature']}: {importance_df.iloc[idx]['Importance']:.2%} impact")
            
            medium_risk = df[df['Risk_Level'] == 'Medium Risk']
            if len(medium_risk) > 0:
                st.warning(f"⚡ {len(medium_risk)} students identified as Medium Risk")
                st.markdown("**Proactive Intervention Strategies:**")
                st.markdown("""
                - 📚 Provide targeted additional practice materials
                - 👨‍👩‍👧‍👦 Organize small group study sessions (3-5 students)
                - 📊 Implement bi-weekly progress monitoring
                - 💡 Offer optional tutoring sessions
                - 🎓 Encourage attendance in extra help sessions
                - 📝 Provide study skills workshops
                """)
                
                with st.expander("📋 View Medium Risk Students"):
                    st.dataframe(medium_risk, use_container_width=True)
            
            low_risk = df[df['Risk_Level'] == 'Low Risk']
            excellent = df[df['Risk_Level'] == 'Excellent']
            
            if len(low_risk) > 0 or len(excellent) > 0:
                st.success(f"✅ {len(low_risk) + len(excellent)} students performing well")
                st.markdown("**Engagement & Enrichment:**")
                st.markdown("""
                - 🌟 Encourage peer teaching and mentorship opportunities
                - 📖 Provide advanced enrichment activities and challenges
                - 🏆 Implement recognition and rewards program
                - 🎯 Set stretch goals for continued growth
                - 💬 Regular positive reinforcement and feedback
                - 🚀 Offer leadership roles in study groups
                """)
                
                if len(excellent) > 0:
                    with st.expander("🌟 View Excellent Performers"):
                        st.dataframe(excellent, use_container_width=True)

# INDIVIDUAL PREDICTION PAGE
elif page == "🎯 Individual Prediction":
    st.header("Individual Student Performance Predictor")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Upload & Predict' page.")
    else:
        st.info("Use this tool to predict performance for a new student based on their current metrics")
        
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            st.subheader("Enter Student Information")
            for i, feature in enumerate(st.session_state.features):
                if i % 2 == 0:
                    # Get reasonable default based on feature name
                    if 'Attendance' in feature:
                        default_val = 85.0
                        min_val, max_val = 0.0, 100.0
                    elif 'Study' in feature or 'Hours' in feature:
                        default_val = 5.0
                        min_val, max_val = 0.0, 24.0
                    elif 'Score' in feature:
                        default_val = 75.0
                        min_val, max_val = 0.0, 100.0
                    elif 'Sleep' in feature:
                        default_val = 7.0
                        min_val, max_val = 0.0, 12.0
                    elif 'Stress' in feature:
                        default_val = 3.0
                        min_val, max_val = 1.0, 10.0
                    else:
                        default_val = 50.0
                        min_val, max_val = 0.0, 100.0
                    
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=0.1,
                        key=f"input_{feature}"
                    )
        
        with col2:
            st.subheader("Enter Student Information (cont.)")
            for i, feature in enumerate(st.session_state.features):
                if i % 2 == 1:
                    if 'Attendance' in feature:
                        default_val = 85.0
                        min_val, max_val = 0.0, 100.0
                    elif 'Study' in feature or 'Hours' in feature:
                        default_val = 5.0
                        min_val, max_val = 0.0, 24.0
                    elif 'Score' in feature:
                        default_val = 75.0
                        min_val, max_val = 0.0, 100.0
                    elif 'Sleep' in feature:
                        default_val = 7.0
                        min_val, max_val = 0.0, 12.0
                    elif 'Stress' in feature:
                        default_val = 3.0
                        min_val, max_val = 1.0, 10.0
                    else:
                        default_val = 50.0
                        min_val, max_val = 0.0, 100.0
                    
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=0.1,
                        key=f"input_{feature}"
                    )
        
        if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = st.session_state.model.predict(input_df)[0]
            
            # Determine risk level
            if prediction < 40:
                risk = "High Risk"
                color = "#ef4444"
                emoji = "🚨"
            elif prediction < 60:
                risk = "Medium Risk"
                color = "#f59e0b"
                emoji = "⚠️"
            elif prediction < 75:
                risk = "Low Risk"
                color = "#10b981"
                emoji = "✅"
            else:
                risk = "Excellent"
                color = "#3b82f6"
                emoji = "🌟"
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {emoji} Risk Level")
                st.markdown(f"<h2 style='color: {color};'>{risk}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Predicted Score")
                st.markdown(f"<h2 style='color: {color};'>{prediction:.1f}</h2>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("### 📈 Percentile")
                if st.session_state.df is not None and 'Predicted_Score' in st.session_state.df.columns:
                    percentile = (st.session_state.df['Predicted_Score'] < prediction).sum() / len(st.session_state.df) * 100
                    st.markdown(f"<h2>{percentile:.0f}th</h2>", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(prediction / 100)
            
            # Recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            if risk == "High Risk":
                st.error("**Immediate Action Required!**")
                st.markdown("""
                - 🆘 Schedule emergency counseling session
                - 📚 Intensive remedial support needed
                - 👨‍🏫 Daily check-ins recommended
                - 📞 Parent/guardian meeting urgent
                - 🎯 Focus on fundamental concepts
                """)
            elif risk == "Medium Risk":
                st.warning("**Proactive Support Recommended**")
                st.markdown("""
                - 📖 Additional practice materials
                - 👥 Join study groups
                - 📅 Weekly progress reviews
                - 💪 Build confidence through small wins
                - 🔍 Identify and address weak areas
                """)
            elif risk == "Low Risk":
                st.success("**Maintain & Strengthen**")
                st.markdown("""
                - ✅ Continue current study habits
                - 📈 Set higher achievement goals
                - 🎓 Explore advanced topics
                - 👥 Consider peer tutoring roles
                - 🌟 Stay motivated and consistent
                """)
            else:
                st.success("**Excellent Performance!**")
                st.markdown("""
                - 🏆 Outstanding work!
                - 🚀 Ready for leadership roles
                - 🎯 Challenge yourself further
                - 👨‍🏫 Mentor other students
                - 📚 Explore enrichment opportunities
                """)
            
            # Feature analysis
            st.markdown("### 📊 Input Analysis")
            
            # Create a comparison with dataset average
            if st.session_state.df is not None:
                comparison_data = []
                for feature in st.session_state.features:
                    student_val = input_data[feature]
                    avg_val = st.session_state.df[feature].mean()
                    comparison_data.append({
                        'Feature': feature,
                        'Your Value': student_val,
                        'Class Average': avg_val,
                        'Difference': student_val - avg_val
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Your Value',
                    x=comparison_df['Feature'],
                    y=comparison_df['Your Value'],
                    marker_color='#6366f1'
                ))
                fig.add_trace(go.Bar(
                    name='Class Average',
                    x=comparison_df['Feature'],
                    y=comparison_df['Class Average'],
                    marker_color='#ec4899'
                ))
                fig.update_layout(
                    title='Your Performance vs Class Average',
                    barmode='group',
                    xaxis_title='Features',
                    yaxis_title='Value'
                )
                st.plotly_chart(fig, use_container_width=True)

# MODEL INFO PAGE
elif page == "ℹ️ Model Info":
    st.header("🤖 About the AI Models")
    
    tab1, tab2, tab3 = st.tabs(["📚 Algorithm Overview", "🔬 How It Works", "📊 Model Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌳 Random Forest Regressor
            
            **Type:** Ensemble Learning Method
            
            **How it works:**
            - Creates multiple decision trees during training
            - Each tree makes a prediction
            - Final prediction is the average of all trees
            - Reduces overfitting through ensemble approach
            
            **Strengths:**
            - ✅ Handles non-linear relationships well
            - ✅ Robust to outliers
            - ✅ Provides feature importance
            - ✅ Works well with small to medium datasets
            - ✅ Minimal hyperparameter tuning needed
            
            **Best For:**
            - Complex patterns in student data
            - When multiple factors interact
            - Datasets with 100+ samples
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Linear Regression
            
            **Type:** Statistical Learning Method
            
            **How it works:**
            - Finds the best straight line through the data
            - Assumes linear relationship between features and target
            - Uses coefficients to weight each feature
            - Minimizes prediction errors mathematically
            
            **Strengths:**
            - ✅ Simple and interpretable
            - ✅ Fast training and prediction
            - ✅ Works well with linear relationships
            - ✅ Requires less data
            - ✅ Shows direct feature impact via coefficients
            
            **Best For:**
            - Clear linear trends
            - When interpretability is crucial
            - Smaller datasets (50+ samples)
            """)
    
    with tab2:
        st.markdown("""
        ### 🔬 The Prediction Process
        
        #### Step 1: Data Preparation
        - Upload student performance data (CSV format)
        - System validates and cleans the data
        - Missing values are handled automatically
        - Features are selected for analysis
        
        #### Step 2: Model Training
        - Data is split into training (80%) and testing (20%) sets
        - Algorithm learns patterns from training data
        - Model identifies relationships between features and performance
        - Cross-validation ensures reliability
        
        #### Step 3: Prediction Generation
        - Trained model analyzes new student data
        - Predicts expected performance score
        - Calculates confidence and accuracy metrics
        - Assigns risk levels based on predictions
        
        #### Step 4: Risk Classification
        - **High Risk**: Predicted score < 40 (Immediate intervention needed)
        - **Medium Risk**: Predicted score 40-60 (Proactive support recommended)
        - **Low Risk**: Predicted score 60-75 (Maintain current performance)
        - **Excellent**: Predicted score > 75 (Outstanding performance)
        
        #### Step 5: Recommendations
        - Personalized action plans generated
        - Feature importance analysis shows key factors
        - Comparative analytics with class averages
        - Downloadable reports for stakeholders
        """)
        
        st.info("💡 **Tip:** The more quality data you provide, the more accurate the predictions become!")
    
    with tab3:
        st.markdown("### 📊 When to Use Each Model")
        
        comparison_df = pd.DataFrame({
            'Criteria': [
                'Dataset Size',
                'Training Speed',
                'Prediction Accuracy',
                'Interpretability',
                'Handles Non-linearity',
                'Overfitting Risk',
                'Feature Importance',
                'Computational Cost'
            ],
            'Random Forest': [
                '100+ samples (ideal)',
                'Slower',
                'Generally Higher',
                'Moderate',
                'Excellent',
                'Low',
                'Built-in',
                'Higher'
            ],
            'Linear Regression': [
                '50+ samples',
                'Very Fast',
                'Good for linear data',
                'Excellent',
                'Poor',
                'Moderate',
                'Via coefficients',
                'Very Low'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Decision Guide
        
        **Choose Random Forest when:**
        - You have sufficient data (100+ students)
        - Relationships are complex and non-linear
        - You need the highest possible accuracy
        - Feature interactions are important
        
        **Choose Linear Regression when:**
        - You have limited data (50-100 students)
        - You need fast predictions
        - Interpretability is critical
        - Relationships appear to be linear
        
        **Performance Metrics Explained:**
        - **R² Score**: Ranges from 0 to 1 (higher is better)
          - > 0.8: Excellent model
          - 0.6-0.8: Good model
          - < 0.6: Consider more data or features
        - **Mean Absolute Error (MAE)**: Average prediction error
        - **Root Mean Squared Error (RMSE)**: Penalizes large errors more
        """)

# ABOUT PAGE
else:
    st.header("About EduMetrics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Project Overview
        
        **EduMetrics: Predict & Progress** is an AI-powered student performance analysis system 
        designed to help educators identify at-risk students early and provide timely interventions.
        
        ### 👥 Development Team
    
    
        - **Parth Tyagi** 
    
        **Academic Year:** 2025-26
        
        ### 🎯 Problem Statement
        
        Educational institutions face several challenges:
        
        - 📉 **Late Detection**: Academic struggles often identified too late
        - 📊 **Data Overload**: Teachers manage too many students to track individually
        - 🎯 **Intervention Gaps**: Lack of data-driven decision making
        - 📈 **Progress Tracking**: Difficulty monitoring improvement over time
        - 👨‍👩‍👧‍👦 **Parent Communication**: Limited real-time performance insights
        
        ### 💡 Our Solution
        
        EduMetrics addresses these challenges through:
        
        1. **Predictive Analytics**: AI-powered performance forecasting
        2. **Early Warning System**: Identify at-risk students proactively
        3. **Data-Driven Insights**: Evidence-based recommendations
        4. **Automated Monitoring**: Efficient tracking for large classes
        5. **Actionable Reports**: Clear next steps for interventions
        
        ### 🌟 Key Features
        
        #### For Teachers
        - 📊 Batch prediction for entire classes
        - 🎯 Risk-level categorization
        - 📈 Performance trend analysis
        - 💾 Exportable reports
        - 🔍 Feature importance insights
        
        #### For Students
        - 📉 Early awareness of weak areas
        - 🎯 Personalized study recommendations
        - 📊 Performance benchmarking
        - 🏆 Goal tracking capabilities
        
        #### For Parents
        - 📱 Timely performance updates
        - 🎯 Clear intervention strategies
        - 📊 Progress monitoring tools
        - 💬 Data-backed discussions
        """)
    
    with col2:
        st.image("https://img.icons8.com/clouds/200/000000/student-center.png", width=200)
        
        st.success("""
        ### 🏆 Impact
        
        **Expected Outcomes:**
        - 30% earlier detection of struggling students
        - 25% improvement in intervention success
        - 50% reduction in manual tracking time
        - Better student-teacher communication
        """)
        
        st.info("""
        ### 💻 Technology Stack
        
        - **Python 3.9+**
        - **Scikit-learn** - ML algorithms
        - **Streamlit** - Web interface
        - **Plotly** - Interactive visualizations
        - **Pandas** - Data processing
        - **NumPy** - Numerical computing
        """)
        
        st.warning("""
        ### 📚 Learning Outcomes
        
        Through this project, we learned:
        - Machine Learning fundamentals
        - Data preprocessing techniques
        - Web application development
        - UI/UX design principles
        - Real-world problem solving
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Future Enhancements
    
    We plan to add:
    - 📱 Mobile application version
    - 🔔 Automated email/SMS alerts
    - 📊 Advanced visualization dashboards
    - 🤖 Deep learning models
    - 📈 Time-series performance tracking
    - 🌐 Multi-language support
    - 🔐 User authentication system
    - 📅 Integration with school management systems
    
    ### 📞 Contact & Feedback
    
    We welcome feedback and suggestions! This project was created as part of the 
    **IBM-CBSE AI Initiative** in collaboration with **Edunet Foundation**.
    
    ### 🙏 Acknowledgments
    
    Special thanks to:

    - **IBM & Edunet Foundation** - AI education initiative
    - **CBSE** - Promoting AI education in schools
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 License")
        st.write("Educational Use Only")
    with col2:
        st.markdown("### 🔖 Version")
        st.write("v2.0.0 (Enhanced)")
    with col3:
        st.markdown("### 📅 Released")
        st.write("January 2026")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #6366f1; font-size: 1.2rem; font-weight: bold;'>
            EduMetrics © 2026
        </p>
        <p style='color: gray;'>
            Built By @tparthhh| Empowering Education Through AI
        </p>
    </div>
    """,
    unsafe_allow_html=True

)
