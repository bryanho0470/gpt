import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage


st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💬"
)
st.title("InvestorGPT")

st.markdown("""
    
    Welcome to InvestorGPT
            
    Write down the name of a company and our Agent will do the research for you.

    """
)
with st.sidebar:
    st.subheader("API setting")

    if 'api_confirmed' not in st.session_state:
        st.session_state.api_confirmed = False

    if not st.session_state["api_confirmed"]:
        openai_api_key = st.text_input("Enter your OpenAI API KEY!", type="password")
        confirm_button = st.button("Confirm API Key")

        if confirm_button:
            if openai_api_key and openai_api_key.startswith("sk-"):
                st.session_state["api_key"] = openai_api_key
                st.session_state["api_confirmed"] = True
                st.success("API key confirmed!")
            else:
                st.error("Invalid API key.")
                st.stop()
        else:
            st.stop()
    else:
        openai_api_key = st.session_state["api_key"]
        st.success("API key confirmed!")
        st.balloons()

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    openai_api_key=openai_api_key,
)

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query : str = Field(
        description="The Query you will search for. Example quesry: Stock Market Symbol for Apple Company"
    )

class StockMarketSymbolSearchTool(BaseTool):
    name : str ="StockMarketSymbolSearchTool"
    description : str = """
    Use this toll to find the stock market symbol for a company.
    It takes a query as an argument.
    """
    args_schema : Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class CompanyOverviewArgsSchema(BaseModel):
    symbol : str = Field(description="Stock symbol of the company. Examples: AAPL, TSLA",)

class CompanFinancialOverviewTool(BaseTool):
    name: str="CompanyOverview"
    description : str = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=Overview&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()

class CompanyIncomeStatementTool(BaseTool):
    name : str = "CompanyIncomeStatement"
    description : str = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get (f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()["annualReports"]

class CompanyStockPerformanceTool(BaseTool):
    name : str = "CompanyStockPerformance"
    description : str = """
    Use this to get the Company stock persformance.
    You should enter a stock symbol
    """
    args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=Time_SERIES_MONTHLY_Adjusted&symbol={symbol}&apikey={alpha_vantage_api_key}")

        return r.json()["Monthly Adjusted Time Series"]



agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanFinancialOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs = {
        "system_message": SystemMessage(content="""
        you are a hedge fund manager.
        You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
                                        
        Consider the performance of a stock, the company overview and the income statement.
                                        
        Be assertive in your judgement and recommend the stock or advice the user against it.
        
        """
        )
    }
)



company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)
    st.write(result["output"])