# **💸 NeoFin: The Risk-Aware Financial Assistant**

**Built for the NeoStats AI Engineer Case Study**

This project transforms a generic chatbot template into a powerful, data-driven financial assistant named "NeoFin". It directly addresses the case study's core challenge: to "define the problem and build the solution" by architecting a smart, usable application from the ground up.

**The Problem:** Most financial advice is generic. It rarely accounts for an individual's specific risk tolerance or dynamically analyzes the market.

**The Solution:** NeoFin is a two-part application that provides financial guidance *dynamically tailored* to a user's stated risk profile, all powered by a blend of LLMs and live, data-driven financial analysis.

## **✨ Core Features**

NeoFin is built on a modular system that integrates all mandatory assessment tasks into a single, cohesive financial application.

### **1\. Mandatory Assessment Features**

* **Retrieval-Augmented Generation (RAG):** Users can upload PDF market reports, which are vectorized using local sentence-transformers. The assistant can then analyze and answer questions about these private documents.  
* **Live Web Search:** Integrated with Tavily to pull real-time news and market commentary, ensuring all advice is current.  
* **Concise vs. Detailed Modes:** A UI toggle to control the verbosity of the AI's responses.

### **2\. Custom Solution: The "NeoFin" Assistant**

This is the core of the project, turning the template into a specialized financial tool. The application is divided into two main pages:

#### **Page 1: The Risk-Aware Chat 💬**

A conversational assistant for day-to-day financial questions.

* **Risk-Profiled Persona:** The user selects their risk tolerance (Low, Medium, or High). This *fundamentally changes* the AI's system prompt, altering its recommendations and tone to match the user.  
* **Multi-Tool Integration:** The chatbot seamlessly combines three tools to answer questions:  
  1. **yfinance:** Pulls live stock data (price, market cap) for any ticker mentioned.  
  2. **Tavily:** Fetches breaking news and analysis.  
  3. **RAG:** Accesses user-uploaded reports for deep-context answers.

#### **Page 2: The Data-Driven Goal Planner 🎯**

This is the app's most powerful feature. It generates a complete, data-driven investment plan from scratch.

**Here's the process:**

1. **User Input:** The user provides their goal (e.g., $1,000,000), investment type (Lumpsum or SIP), and risk profile.  
2. **Tenure Calculation:** The app first calculates the *estimated time* to reach the goal based on risk-adjusted return expectations.  
3. **Dynamic Asset Analysis (The Core Logic):**  
   * **Analyze:** The app downloads **15 years** of historical data for a 28-asset universe (ETFs covering Stocks, Bonds, Real Estate, Commodities, and Crypto).  
   * **Calculate:** It computes the **Annualized Return** (profit) and **Annualized Volatility** (risk) for every single asset.  
   * **Filter:** It *dynamically selects* the top 5 assets that mathematically match the user's risk profile (e.g., "Low Risk" \= lowest volatility; "Medium Risk" \= best risk-adjusted return).  
4. **Synthesize & Recommend:**  
   * The app feeds this hard data (15-year performance, live prices, recent news) into the Groq LLM as context.  
   * The LLM then acts as an expert financial planner, using this data to build a custom-tailored investment basket and asset allocation for the user.

## **🛠️ Tech Stack**

* **Frontend:** Streamlit  
* **LLM & Orchestration:** LangChain, Groq (for high-speed LLM inference)  
* **Data Analysis:** Pandas, Numpy  
* **Financial Data:** yfinance (for historical & live stock/ETF/crypto data)  
* **Web Search:** Tavily  
* **Vector DB (RAG):** FAISS, sentence-transformers (for local, free embeddings)

## **📂 Project Structure**

The project follows the modular structure specified in the assessment:

```
AI\_UseCase/  
│  
├── .env                  \# Stores API keys  
├── app.py                \# Main Streamlit UI logic (router & pages)  
├── requirements.txt      \# All dependencies  
│  
├── config/  
│   └── config.py         \# Loads API keys from .env  
│  
├── models/  
│   ├── llm.py            \# Initializes the Groq chat model  
│   └── embeddings.py     \# Initializes the local embedding model  
│  
└── utils/  
    ├── rag\_helper.py       \# Handles PDF parsing and vector store creation  
    ├── search\_helper.py    \# Initializes the Tavily search tool  
    ├── finance\_helper.py   \# Tool for fetching live yfinance data  
    └── goal\_helper.py      \# Core logic for the Goal Planner (analysis & LLM call)
```
## **🚀 How to Run**

1. **Clone the Repository**  
   git clone \[https://github.com/your-username/neofin-project.git\](https://github.com/your-username/neofin-project.git)  
   cd neofin-project

2. **Create and Activate Virtual Environment**  
   python3 \-m venv venv  
   source venv/bin/activate

3. **Install Dependencies**  
   pip install \-r requirements.txt

4. Create Your .env File  
   Create a file named .env in the root directory and add your API keys.  
   \# .env  
   GROQ\_API\_KEY="gsk\_..."  
   TAVILY\_API\_KEY="tvly-..."

   *(Note: OpenAI/Google keys are not required as we use a free, local embedding model).*  
5. **Run the App\!**  
   streamlit run app.py  
