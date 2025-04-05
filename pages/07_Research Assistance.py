import streamlit as st
import requests
from typing import Type
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.schema import SystemMessage


st.set_page_config(
    page_title="Research Assiatance Assignment8",
    page_icon="🤖"
)

st.title("Research Assistance")

st.markdown(
    """
       
    Welcome to Assistance
            
    Write down the key words what you want to know.

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

# === Tool 1: Wikipedia Search ===
class WikiSearchArgs(BaseModel):
    query: str = Field(description="Search term for Wikipedia")

class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearch"
    description = "Search Wikipedia and return relevant summary."
    args_schema: Type[WikiSearchArgs] = WikiSearchArgs

    def _run(self, query: str):
        wiki = WikipediaAPIWrapper()
        return wiki.run(query)

# === Tool 2: DuckDuckGo Search ===
class DuckSearchArgs(BaseModel):
    query: str = Field(description="Search query for DuckDuckGo")

class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearch"
    description = "Search DuckDuckGo and return URLs or snippets."
    args_schema: Type[DuckSearchArgs] = DuckSearchArgs

    def _run(self, query: str):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)

# === Tool 3: Website Scraper ===
class ScrapeWebsiteArgs(BaseModel):
    url: str = Field(description="A full URL of the website to scrape")

class WebsiteScraperTool(BaseTool):
    name = "WebsiteScraper"
    description = "Extract visible text from a given webpage URL"
    args_schema: Type[ScrapeWebsiteArgs] = ScrapeWebsiteArgs

    def _run(self, url: str):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            return text[:5000]  # limit text to prevent token overflow
        except Exception as e:
            return f"Error scraping {url}: {e}"

# === Tool 4: Save to File ===
class SaveFileArgs(BaseModel):
    content: str = Field(description="Text content to save")

class SaveResearchTool(BaseTool):
    name = "SaveResearch"
    description = "Save research content to a .txt file"
    args_schema: Type[SaveFileArgs] = SaveFileArgs

    def _run(self, content: str):
        path = "xz_backdoor_research.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Research saved to {path}"
    
agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        WikipediaSearchTool(),
        DuckDuckGoSearchTool(),
        WebsiteScraperTool(),
        SaveResearchTool()
    ],
    agent_kwargs = {
        "system_message": SystemMessage(content=
        """
        You are very brilliant research assistant. You can search for information on Wikipedia, DuckDuckGo, and scrape websites.
        You can also save the information you find to a .txt file.
        """
        )
    }
)



company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)
    st.write(result["output"])