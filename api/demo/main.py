import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data_model import InputPrompt
from autosql_demo import engine, metadata_obj
from autosql_demo import initialise_llm, initialise_db_chain, initialise_tools
from autosql_demo import initialise_zeroshot_agent, initialise_conversational_agent

VERBOSE_FLAG = True
app = FastAPI() # Create FastAPI app
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = initialise_llm() # Initialize the LLM
sql_chain = initialise_db_chain(llm, verbose_flag=VERBOSE_FLAG) # Initialize the database chain
tools = initialise_tools(llm, sql_chain) # Initialize the tools
zeroshot_agent = initialise_zeroshot_agent(llm, tools, verbose_flag=VERBOSE_FLAG) # Initialize the zeroshot agent
conversational_agent = initialise_conversational_agent(llm, tools, verbose_flag=VERBOSE_FLAG) # Initialize the conversational agent

# sql_chain.run("Generate SQL query to fetch what is the company name for stock ticker ABC")
# zeroshot_agent.run("Generate SQL query to fetch what is the company name for stock ticker ABC")
# conversational_agent.run("Generate SQL query to fetch what is the company name for stock ticker ABC")

@app.post("/autosql_sqlchain")
def autosql_sqlchain(prompt: InputPrompt) -> str:
    """
    AutoSQL SQLchain endpoint

    @param user_query_english: User's query in English
    @return: SQL query result from LLM
    """
    user_english_query = prompt.user_english_query
    result = sql_chain.run(user_english_query)
    return result

@app.post("/autosql_zeroshot")
def autosql_zeroshot(prompt: InputPrompt) -> str:
    """
    AutoSQL zero shot agent endpoint

    @param user_query_english: User's query in English
    @return: SQL query result from LLM
    """
    user_english_query = prompt.user_english_query
    zeroshot_agent = initialise_zeroshot_agent(llm, tools, verbose_flag=VERBOSE_FLAG) # Initialize the zeroshot agent
    result = zeroshot_agent.run(user_english_query)
    return result

@app.post("/autosql_conversational")
def autosql_conversational(prompt: InputPrompt) -> str:
    """
    AutoSQL conversational agent endpoint

    @param user_query_english: User's query in English
    @return: SQL query result from LLM
    """
    user_english_query = prompt.user_english_query
    result = conversational_agent.run(user_english_query)
    return result




    