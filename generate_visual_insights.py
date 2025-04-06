import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_groq import ChatGroq
from pandasai import SmartDataframe
from pandasai.responses.streamlit_response import StreamlitResponse

class GenerateVisualInsights:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model_name="llama3-70b-8192", api_key=self.api_key)
        
    def chat_with_dataframe(self, smart_df, prompt):
        try:
            result = smart_df.chat(prompt)
            return result
        except Exception as e:
            st.error(f"❌ Error while querying: {e}")
            return None
    
    def view(self, model=None, ui_width=None, device_type=None, device_width=None):
        st.set_page_config(layout="wide")
        st.title("📊 Generate Visual & Textual Insights with Groq + Mixtral")
        
        uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    # Try UTF-8 first, then fallback
                    try:
                        data = pd.read_csv(uploaded_file)
                    except UnicodeDecodeError:
                        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                    st.success("✅ CSV Uploaded Successfully")
                elif file_extension in ['xlsx', 'xls']:
                    data = pd.read_excel(uploaded_file)
                    st.success("✅ Excel File Uploaded Successfully")
                else:
                    st.error("❌ Unsupported file format")
                    return
                
                st.dataframe(data, use_container_width=True)
                
                # Add tabs for visualization and text query
                tab1, tab2 = st.tabs(["Visual Insights", "Text Analysis"])
                
                with tab1:
                    visual_query = st.text_area(
                        "💬 Ask for visual insights", 
                        placeholder="E.g. Show total sales by country as a bar chart"
                    )
                    
                    if visual_query and st.button("🚀 Generate Visualization", key="visual_btn"):
                        st.info(f"🔍 Your Query: {visual_query}")
                        smart_df = SmartDataframe(data, config={
                            "llm": self.llm,
                            "verbose": True,
                            "response_parser": StreamlitResponse,
                            "enable_cache": False,
                            "save_charts": False,
                            "save_code": True
                        })
                        result = self.chat_with_dataframe(smart_df, visual_query)
                        self._display_result(result, smart_df)
                
                with tab2:
                    text_query = st.text_area(
                        "💬 Ask for text-based analysis", 
                        placeholder="E.g. What's the average sales value? Summarize key trends in the data."
                    )
                    
                    if text_query and st.button("🚀 Get Text Analysis", key="text_btn"):
                        st.info(f"🔍 Your Query: {text_query}")
                        smart_df = SmartDataframe(data, config={
                            "llm": self.llm,
                            "verbose": True,
                            "response_parser": None,  # Use default text response
                            "enable_cache": False,
                            "save_code": True
                        })
                        result = self.chat_with_dataframe(smart_df, text_query)
                        self._display_result(result, smart_df)
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
        else:
            st.info("📌 Please upload a CSV or Excel file to start.")
    
    def _display_result(self, result, smart_df):
        """Helper method to display results consistently"""
        if result is None:
            st.warning("⚠️ No result returned.")
        elif isinstance(result, pd.DataFrame):
            st.write("📄 Generated DataFrame:")
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, str):
            st.write("🧠 Answer:")
            st.success(result)
        else:
            st.write("ℹ️ Generated Output:")
            st.write(result)
            
        if smart_df.last_code_executed:
            with st.expander("🧠 Code used by the model"):
                st.code(smart_df.last_code_executed, language="python")