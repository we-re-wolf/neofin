# app.py
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config.config import settings
from models.llm import get_chatgroq_model
from models.embeddings import get_openai_embeddings
from utils.rag_helper import get_pdf_text, get_text_chunks, get_vector_store
from utils.search_helper import get_web_search_tool
from utils.finance_helper import get_stock_data
from utils.goal_helper import calculate_tenure, get_investment_basket

def get_chat_response(chat_model, messages, system_prompt, retriever, use_web_search, use_stock_data, response_mode):
    """Get response from the chat model, integrating RAG, Web Search, and Finance Tools."""
    try:
        full_system_prompt = system_prompt

        if response_mode == "Concise":
            full_system_prompt += "\n\nProvide a short, summarized reply (Concise Mode)."
        else:
            full_system_prompt += "\n\nProvide an expanded, in-depth response (Detailed Mode)."

        formatted_messages = [SystemMessage(content=full_system_prompt)]

        last_user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        context_str = ""

        if retriever:
            try:
                relevant_docs = retriever.invoke(last_user_message)
                context_str += "--- START (Knowledge Base Context) ---\n"
                for i, doc in enumerate(relevant_docs):
                    context_str += f"Source {i+1}:\n{doc.page_content}\n\n"
                context_str += "--- END (Knowledge Base Context) ---\n"
            except Exception as e:
                st.error(f"Error during RAG retrieval: {e}")

        if use_web_search:
            try:
                search_tool = get_web_search_tool()
                if search_tool:
                    search_results = search_tool.invoke(last_user_message)
                    context_str += f"--- START (Live Web Search Results) ---\n{search_results}\n--- END (Live Web Search Results) ---\n"
                else:
                    st.warning("Web search is enabled, but TAVILY_API_KEY is not configured.")
            except Exception as e:
                st.error(f"Error during web search: {e}")

        if use_stock_data:
            try:
                import re
                potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', last_user_message)
                if potential_tickers:
                    for ticker in potential_tickers:
                        stock_data = get_stock_data.invoke(ticker)
                        context_str += f"--- START (Live Stock Data: {ticker}) ---\n{stock_data}\n--- END (Live Stock Data: {ticker}) ---\n"
            except Exception as e:
                st.error(f"Error getting stock data: {e}")

        if context_str:
            full_system_prompt += f"\n\nUse the following context to answer the user's question:\n{context_str}"
            formatted_messages[0] = SystemMessage(content=full_system_prompt)

        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## Installation
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    Create a file named `.env` in the main `AI_UseCase` folder:

    ```
    # .env file
    GROQ_API_KEY="gsk_..."
    TAVILY_API_KEY="tvly-..."
    # We are now using local embeddings, so OpenAI/Google keys are not needed for RAG.
    ```
    
    ## How to Use
    
    1. **Go to the Chat page**
    2. **Set your Risk Profile** in the sidebar.
    3. **Start chatting!** Ask for stock info (e.g., "Tell me about AAPL") or recommendations.
                
    ## Few things about the application
    
    1. **We are not using any openai/gemini api key as we are getting rate limits for embeddings**
    2. **You can go to the personal goals page and get stock recommendation** in the sidebar.
    3. **Just like groww you will get asset allocation and time to achieve it!** We are currently using 27 etfs as other wise we would face rate limits and streamlit would crash as well.
    """)

def chat_page(chat_model, embeddings_model):
    """Main chat interface page"""
    st.title("💸 NeoFin Assistant")

    web_search_tool = get_web_search_tool()

    if not chat_model:
        st.error("**Missing API Key:** GROQ_API_KEY (for chat). Please check the 'Instructions' page.")
        return 

    with st.sidebar:
        st.header("User Configuration")

        st.subheader("Your Risk Profile")
        risk_profile = st.selectbox(
            "Select your investment risk tolerance:",
            ("Low Risk", "Medium Risk", "High Risk"),
            index=1 
        )
        st.info(f"All recommendations will be tailored for a **{risk_profile}** tolerance.")

        st.divider()
        
        st.header("Chat Tools")

        st.subheader("Web Search")
        use_web_search = st.toggle("Enable Live Web Search", value=True)
        if use_web_search and not web_search_tool:
            st.warning("Web search enabled, but TAVILY_API_KEY is missing.")

        st.subheader("Stock Data")
        use_stock_data = st.toggle("Enable Live Stock Data", value=True)

        st.divider()

        st.subheader("Response Mode")
        response_mode = st.radio(
            "Select response detail level:",
            ["Concise", "Detailed"],
            index=1 
        )

        st.divider()

        st.subheader("Knowledge Base (RAG)")
        uploaded_files = st.file_uploader(
            "Upload market reports (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        raw_text = get_pdf_text(uploaded_files)
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.retriever = get_vector_store(text_chunks, embeddings_model)
                        st.success(f"Knowledge Base built from {len(uploaded_files)} file(s)!")
                    except Exception as e:
                        st.error(f"Failed to build Knowledge Base: {e}")
            else:
                st.warning("Please upload at least one PDF file.")

        if "retriever" in st.session_state:
            st.success("Knowledge Base is active.")
            if st.button("Clear Knowledge Base"):
                del st.session_state.retriever
                st.rerun()

    system_prompt = f"""
    You are a professional financial assistant. Your name is "NeoFin".
    You MUST provide advice and recommendations that are strictly tailored to the user's risk profile.
    The user's current risk profile is: **{risk_profile}**.

    - For **Low Risk** users: Prioritize capital preservation. Recommend stable, large-cap stocks (blue-chip), bonds, and diversified ETFs. Be very cautious about speculation.
    - For **Medium Risk** users: Recommend a balanced portfolio, including growth stocks, index funds, and some allocation to more stable assets.
    - For **High Risk** users: You can discuss more speculative assets, growth stocks, and smaller-cap companies, but ALWAYS remind them of the high risk involved.
    
    **CRITICAL_RULE**: You must NEVER give a "buy" or "sell" recommendation. Instead of "you should buy AAPL," say "AAPL is a strong company that aligns with your risk profile because..."
    ALWAYS be helpful, professional, and provide a disclaimer that you are an AI assistant and this is not financial advice.
    """
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about markets or stocks (e.g., 'What do you think of TSLA?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                retriever = st.session_state.get("retriever")
                
                response = get_chat_response(
                    chat_model, 
                    st.session_state.messages, 
                    system_prompt,
                    retriever,
                    use_web_search,
                    use_stock_data,
                    response_mode
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def personal_goals_page(chat_model):
    """Page for calculating financial goals and getting a basket."""
    st.title("🎯 Personal Financial Goals")

    expected_returns = {
        "Low Risk": 0.08,     
        "Medium Risk": 0.10,  
        "High Risk": 0.14     
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Goal")
        goal_amount = st.number_input("What is your target amount ($)?", min_value=1000.0, value=1000000.0, step=10000.0, format="%.2f")
        risk_profile = st.selectbox(
            "What is your risk profile?",
            ("Low Risk", "Medium Risk", "High Risk"),
            index=1
        )
        st.info(f"We'll use an estimated annual return of **{expected_returns[risk_profile]*100:.0f}%** based on your risk profile.")

    with col2:
        st.subheader("Your Investment Plan")
        investment_type = st.radio("Investment Type", ["SIP", "Lumpsum"])
        
        is_step_up = False
        step_up_percent = 0.0
        
        if investment_type == "SIP":
            amount = st.number_input("Monthly SIP Amount ($)", min_value=10.0, value=500.0, step=50.0)
            is_step_up = st.checkbox("Enable Step-up SIP? (Annual Increase)")
            if is_step_up:
                step_up_percent = st.slider("Annual Step-up Percentage", min_value=1, max_value=20, value=10, format="%d%%")
        else:
            amount = st.number_input("Lumpsum Amount ($)", min_value=100.0, value=25000.0, step=100.0)

    st.divider()

    if st.button("Build My Plan", use_container_width=True, type="primary"):
        if not chat_model:
            st.error("Chat model is not loaded. Please check your GROQ_API_KEY.")
            return

        with st.spinner("Calculating tenure and building your basket..."):
            annual_rate = expected_returns[risk_profile]
            tenure_years = calculate_tenure(goal_amount, investment_type, amount, annual_rate, is_step_up, step_up_percent)
            
            st.subheader(f"Step 1: Estimated Time Horizon")
            if isinstance(tenure_years, str) and tenure_years == "Error":
                st.error("Could not calculate tenure. Your goal may be unreachable with these inputs.")
            elif isinstance(tenure_years, str):
                st.warning(f"Your goal will take **over {tenure_years} years** to reach with this plan.")
            else:
                st.success(f"It will take approximately **{tenure_years} years** to reach your goal of ${goal_amount:,.2f}.")

            if not isinstance(tenure_years, str):
                st.subheader(f"Step 2: Suggested Investment Basket")
                basket_recommendation = get_investment_basket(
                    chat_model, 
                    goal_amount, 
                    risk_profile, 
                    investment_type, 
                    amount, 
                    tenure_years
                )
                st.markdown(basket_recommendation)
            else:
                st.info("A basket could not be generated as the tenure calculation was not successful.")


def main():
    st.set_page_config(
        page_title="NeoFin Financial Assistant",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        chat_model = get_chatgroq_model()
        embeddings_model = get_openai_embeddings()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        chat_model, embeddings_model = None, None

    with st.sidebar:
        st.title("NeoFin Assistant")
        page = st.radio(
            "Go to:",
            ["Chat", "Personal Goals", "Instructions"],
            index=0
        )

        if page == "Chat":
            st.divider()
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page(chat_model, embeddings_model) 
    if page == "Personal Goals":
        personal_goals_page(chat_model) 

if __name__ == "__main__":
    main()