import os
import sys
import datetime
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow import BondWorkflowChain
from src.agents.bond_calculator_agent import BondCalculatorAgent

# App configuration
st.set_page_config(
    page_title="ChatBond.ai",
    page_icon="💰",
    layout="wide"
)

# Initialize the classes
if "workflow_chain" not in st.session_state:
    st.session_state.workflow_chain = BondWorkflowChain()
if "calculator" not in st.session_state:
    st.session_state.calculator = BondCalculatorAgent(
        current_date="2025-03-10 03:14:53",
        current_user="SRINJOY59"
    )
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to ChatBond.ai, SRINJOY59! I can help you find information about bonds, compare yields, analyze cash flows, and screen potential investments."}
    ]

# Header
st.title("ChatBond.ai")
st.write("User: SRINJOY59 | Date: 2025-03-10 03:14:53")

# Create two simple tabs
tab1, tab2 = st.tabs(["Chat", "Bond Calculator"])

# Tab 1: Chat Interface
with tab1:
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.write(f"🤖 **Assistant**: {message['content']}")
        else:
            st.write(f"👤 **You**: {message['content']}")
    
    # Sample queries for quick selection
    st.write("### Quick Questions:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What information do you have on corporate bonds with AAA rating?"):
            st.session_state.user_query = "What information do you have on corporate bonds with AAA rating?"
        if st.button("When is the next payment date for US Treasury bonds?"):
            st.session_state.user_query = "When is the next payment date for US Treasury bonds?"
    with col2:
        if st.button("Show me the highest yielding bonds available right now."):
            st.session_state.user_query = "Show me the highest yielding bonds available right now."
        if st.button("Which companies have the strongest financial metrics in their bond offerings?"):
            st.session_state.user_query = "Which companies have the strongest financial metrics in their bond offerings?"
    
    # User input
    user_query = st.text_input("Ask about bonds:", key="user_query")
    
    # Process user query
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Process the query
        with st.spinner("Getting answer..."):
            try:
                # Direct call to the workflow chain
                response = st.session_state.workflow_chain.process_query(user_query)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # Clear the input and refresh

# Tab 2: Bond Calculator
with tab2:
    st.write("### Bond Yield Calculator")
    st.write("Enter a bond ISIN to calculate its yield:")
    
    # Simple ISIN input
    isin = st.text_input("Bond ISIN:", placeholder="Example: US912810TD00")
    
    if st.button("Calculate Yield"):
        if not isin:
            st.error("Please enter an ISIN")
        else:
            with st.spinner("Calculating..."):
                try:
                    # Direct call to the calculator with just the ISIN
                    result = st.session_state.calculator.process_query(isin)
                    st.success(f"Calculation Result: {result}")
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")