import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import io
import re

# Page config
st.set_page_config(
    page_title="Data Science Co-Pilot", 
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = duckdb.connect(':memory:')
if 'current_table' not in st.session_state:
    st.session_state.current_table = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

class DataScienceCoPilot:
    def __init__(self):
        self.db = st.session_state.db_connection
        
    def load_csv_to_duckdb(self, uploaded_file, table_name: str = "main_table"):
        """Load CSV file directly into DuckDB"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Clean table name
            clean_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
            
            # Create table in DuckDB
            self.db.execute(f"DROP TABLE IF EXISTS {clean_table_name}")
            self.db.execute(f"CREATE TABLE {clean_table_name} AS SELECT * FROM df")
            
            st.session_state.current_table = clean_table_name
            st.session_state.data_loaded = True
            
            return df, clean_table_name
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None, None
    
    def execute_sql_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query on DuckDB"""
        try:
            result = self.db.execute(query).fetchdf()
            return result
        except Exception as e:
            st.error(f"SQL Error: {str(e)}")
            return None
    
    def natural_language_to_sql(self, question: str, table_name: str) -> str:
        """Convert natural language to SQL query"""
        question_lower = question.lower()
        
        # Get table schema
        schema_query = f"DESCRIBE {table_name}"
        try:
            schema_df = self.db.execute(schema_query).fetchdf()
            columns = schema_df['column_name'].tolist()
        except:
            columns = []
        
        # Basic patterns for SQL generation
        if any(word in question_lower for word in ['count', 'how many', 'total']):
            if 'survive' in question_lower and 'Survived' in columns:
                return f"SELECT Survived, COUNT(*) as count FROM {table_name} GROUP BY Survived"
            return f"SELECT COUNT(*) as total_rows FROM {table_name}"
        
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            numeric_cols = self._get_numeric_columns(table_name)
            if 'age' in question_lower and 'Age' in columns:
                if 'class' in question_lower and 'Pclass' in columns:
                    return f"SELECT Pclass, AVG(Age) as avg_age FROM {table_name} GROUP BY Pclass"
                return f"SELECT AVG(Age) as average_age FROM {table_name}"
            elif numeric_cols:
                avg_cols = [f"AVG({col}) as avg_{col}" for col in numeric_cols[:3]]
                return f"SELECT {', '.join(avg_cols)} FROM {table_name}"
        
        elif any(word in question_lower for word in ['missing', 'null', 'empty']):
            null_checks = []
            for col in columns[:10]:  # Limit to first 10 columns
                null_checks.append(f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as {col}_nulls")
            return f"SELECT {', '.join(null_checks)} FROM {table_name}"
        
        elif 'survival rate' in question_lower:
            if 'Sex' in columns and 'Survived' in columns:
                return f"""
                SELECT Sex, 
                       COUNT(*) as total,
                       SUM(Survived) as survived,
                       ROUND(AVG(Survived) * 100, 2) as survival_rate_percent
                FROM {table_name} 
                GROUP BY Sex
                """
        
        elif 'unique' in question_lower:
            # Find mentioned column
            mentioned_col = None
            for col in columns:
                if col.lower() in question_lower:
                    mentioned_col = col
                    break
            if mentioned_col:
                return f"SELECT {mentioned_col}, COUNT(*) as count FROM {table_name} GROUP BY {mentioned_col} ORDER BY count DESC"
        
        elif any(word in question_lower for word in ['show', 'display', 'view']):
            if 'first' in question_lower or 'top' in question_lower:
                return f"SELECT * FROM {table_name} LIMIT 10"
            # Check for specific columns mentioned
            mentioned_cols = [col for col in columns if col.lower() in question_lower]
            if mentioned_cols:
                return f"SELECT {', '.join(mentioned_cols)} FROM {table_name} LIMIT 10"
        
        elif 'schema' in question_lower or 'structure' in question_lower:
            return f"DESCRIBE {table_name}"
        
        # Default: show sample data
        return f"SELECT * FROM {table_name} LIMIT 5"
    
    def _get_numeric_columns(self, table_name: str) -> list:
        """Get numeric columns from table"""
        try:
            schema_df = self.db.execute(f"DESCRIBE {table_name}").fetchdf()
            numeric_types = ['INTEGER', 'BIGINT', 'DOUBLE', 'REAL', 'DECIMAL']
            numeric_cols = schema_df[schema_df['column_type'].isin(numeric_types)]['column_name'].tolist()
            return numeric_cols
        except:
            return []
    
    def get_data_summary(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        try:
            # Basic info
            count_query = f"SELECT COUNT(*) as total_rows FROM {table_name}"
            total_rows = self.db.execute(count_query).fetchone()[0]
            
            # Schema info
            schema_df = self.db.execute(f"DESCRIBE {table_name}").fetchdf()
            
            # Sample data
            sample_df = self.db.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
            
            return {
                'total_rows': total_rows,
                'total_columns': len(schema_df),
                'columns': schema_df['column_name'].tolist(),
                'column_types': dict(zip(schema_df['column_name'], schema_df['column_type'])),
                'sample_data': sample_df
            }
        except Exception as e:
            st.error(f"Error getting summary: {str(e)}")
            return {}

def main():
    # Initialize the co-pilot
    copilot = DataScienceCoPilot()
    
    # Header
    st.title("🤖 Data Science Co-Pilot")
    st.markdown("*Your AI assistant for data analysis - GPU-free and fast with DuckDB!*")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file to start analyzing your data"
        )
        
        if uploaded_file is not None:
            # Get table name from filename
            table_name = uploaded_file.name.replace('.csv', '')
            
            # Load data
            with st.spinner("Loading data into DuckDB..."):
                df, clean_table_name = copilot.load_csv_to_duckdb(uploaded_file, table_name)
            
            if df is not None:
                # Show data summary
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Dataset overview
                st.subheader("📊 Dataset Overview")
                summary = copilot.get_data_summary(clean_table_name)
                
                if summary:
                    st.metric("Rows", summary['total_rows'])
                    st.metric("Columns", summary['total_columns'])
                    
                    with st.expander("Column Information"):
                        for col, dtype in summary['column_types'].items():
                            st.write(f"**{col}**: {dtype}")
    
    # Main chat interface
    if st.session_state.data_loaded:
        st.header("💬 Chat with your Data")
        
        # Quick action buttons
        st.subheader("Quick Questions:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Data Summary"):
                query = "Show me a summary of the data"
                st.session_state.chat_history.append(("user", query))
        
        with col2:
            if st.button("🔍 Missing Values"):
                query = "Find missing values in the dataset"
                st.session_state.chat_history.append(("user", query))
        
        with col3:
            if st.button("📊 Basic Stats"):
                query = "Show basic statistics"
                st.session_state.chat_history.append(("user", query))
        
        with col4:
            if st.button("🏗️ Schema Info"):
                query = "Show the schema and column information"
                st.session_state.chat_history.append(("user", query))
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., 'How many passengers survived?', 'Show average age by class'"
        )
        
        if st.button("🔍 Analyze") and user_question:
            st.session_state.chat_history.append(("user", user_question))
        
        # Process and display chat
        if st.session_state.chat_history:
            # Get the latest question
            latest_question = st.session_state.chat_history[-1]
            
            if latest_question[0] == "user":
                question = latest_question[1]
                
                # Generate SQL query
                sql_query = copilot.natural_language_to_sql(question, st.session_state.current_table)
                
                # Execute query
                with st.spinner("Running analysis..."):
                    result_df = copilot.execute_sql_query(sql_query)
                
                if result_df is not None:
                    # Display results
                    st.subheader("🤖 Analysis Results")
                    
                    # Show the question
                    st.write(f"**Your Question:** {question}")
                    
                    # Show generated SQL
                    with st.expander("Generated SQL Query"):
                        st.code(sql_query, language='sql')
                    
                    # Show results
                    if len(result_df) > 0:
                        st.write("**Results:**")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Auto-visualization for certain types of results
                        if len(result_df.columns) == 2 and len(result_df) <= 20:
                            # Try to create a chart for 2-column results
                            try:
                                col1, col2 = result_df.columns
                                if result_df[col2].dtype in ['int64', 'float64']:
                                    fig = px.bar(result_df, x=col1, y=col2, title=f"{col2} by {col1}")
                                    st.plotly_chart(fig, use_container_width=True)
                            except:
                                pass
                        
                        # Download option
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No results found for your query.")
                
                # Add to chat history
                st.session_state.chat_history.append(("assistant", "Query completed"))
        
        # View Current Dataset
        if st.button("🔍 View Current Dataset"):
            st.subheader("Current Dataset Sample")
            sample_query = f"SELECT * FROM {st.session_state.current_table} LIMIT 10"
            sample_df = copilot.execute_sql_query(sample_query)
            if sample_df is not None:
                st.dataframe(sample_df, use_container_width=True)
    
    else:
        # Welcome message
        st.info("👋 Welcome! Upload a CSV file in the sidebar to start analyzing your data with DuckDB-powered queries.")
        
        # Example queries
        st.subheader("Example Questions You Can Ask:")
        st.markdown("""
        - How many rows are in the dataset?
        - Show me the first 10 rows
        - What are the missing values in each column?
        - Show unique values in a specific column
        - Calculate average values by category
        - What's the survival rate by gender? (for Titanic dataset)
        - Show me basic statistics
        """)

if __name__ == "__main__":
    main()