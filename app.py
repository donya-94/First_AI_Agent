import os
import certifi
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub # this uses to set the prompt by there is an issue and information about it in agent_demo file

from langchain.tools import tool

import requests # this uses to custom tool there is a tool function in agent_demo file

from langchain.agents import create_react_agent, AgentExecutor #react: reasoning and action


#---------------------------------------------------------
# First of all we need to load the variables in .env file
#---------------------------------------------------------

os.environ["SSL_CERT_FILE"] = certifi.where() #SSL certificate configuration
# os.environ: Allows Python to access and modify environment variables.
# "SSL_CERT_FILE": An environment variable that tells some Python libraries where to find trusted SSL certificates.
# What is SSL/TSL? An SSL/TLS certificate helps verify the identity of a website or server and supports encrypted communication.
# certifi.where(): Returns the location of a certificate bundle provided by the certifi Python package.
load_dotenv()#It reads a file named .env in your project and loads its variables into Python's environment.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")


search_tool = TavilySearchResults(max_results= 2) #max_result : how many result we want


#----------------------------------
# Initialize our LLM
#----------------------------------

#----------BECAUSE THE OPENAI API KEY MUST BE PAID i HAVE TO USE GEMINI LLM
# llm = ChatOpenAI(
#     model= "gpt-4.1-mini",
#     temperature= 0,
#     api_key= OPENAI_API_KEY
# )

#--------I had to run this command in terminal: pip install -U langchain-google-genai
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key= GEMINI_API_KEY,
    max_retries=5,       # Automatically retry up to 5 times on 503/429 errors
    timeout=30.0         # Fail fast and retry if a single request takes > 30 seconds
)


#--------------------------------------
# Prompt, we can write prompts mannualy or download it from this website: hwchase17/react
#--------------------------------------

#------- Because of the version of the tutorial and installed on my vscode, I could not use the command below,
#------- and I had to import client directly from langsmith then use its prompts
# prompt = hub.pull("hwchase17/react")# hwchase17/react : is a website 

from langsmith import Client

# Initialize the LangSmith client
client = Client()

# Pull the public prompt with explicit security acknowledgment
prompt = client.pull_prompt(
    "hwchase17/react"
)

# -----------------------------------------
# TOOLS
# -----------------------------------------

tools = [search_tool]# if we set an empty tool: tools = [] then the agent-executer try to find a tool to execut the input that we have given to it, 
#but it will find nothing and it will take into a loop

#-------------------------------------------
# Create Agent
#-------------------------------------------

agent = create_react_agent(
    llm= llm,
    tools= tools, # this is used to external search operation, external refrence
    prompt= prompt
)

#-------------------------------------------
# Executer
#-------------------------------------------

agent_executor = AgentExecutor(
    agent= agent,
    tools= tools,
    verbose= True #we will be able to see the log of anything that agent executes. false makes us unavaialable to see the log
)

# -------------------------------------------
# UI INPUT
#-------------------------------------------

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)

#---------------------------------------
# RUN
#---------------------------------------


if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")