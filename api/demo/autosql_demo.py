import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import Date
from sqlalchemy import ForeignKey
from sqlalchemy import create_engine
from sqlalchemy import insert
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy.schema import CreateTable

from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory


engine = create_engine(os.environ['PG_DATABASE_URL'])
metadata_obj = MetaData()
# engine = create_engine("sqlite:///:memory:", echo=True) # Create a in memory SQLlite database engine
load_dotenv() # Load environment variables from .env file

def initialise_llm():
    """
    Initialize the LLM with the OpenAI API key and the model name, here we are using the davinci model
    """
    llm = OpenAI(
        model_name='text-davinci-003',
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0
    )

    return llm


def initialise_db_chain(llm, verbose_flag=False):
    """
    Initialize the database chain for tables
    """
    metadata_obj.reflect(bind=engine)
    metadata_obj.drop_all(bind=engine)

    stocks = Table(
        "stocks",
        metadata_obj,
        Column("obs_id", Integer, primary_key=True),
        Column("stock_ticker", String(4), nullable=False),
        Column("price", Float, nullable=False),
        Column("date", Date, nullable=False),
        extend_existing=True,
    )

    company = Table(
        "company",
        metadata_obj,
        Column("company_id", Integer, primary_key=True),
        Column("company_name", String(16), nullable=False),
        Column("stock_ticker", String(4), nullable=False),
        extend_existing=True,
    )

    sales = Table(
        "sales",
        metadata_obj,
        Column("sales_id", Integer, primary_key=True),
        Column("company_id", Integer, ForeignKey("company.company_id"), nullable=False),
        Column("date", Date, nullable=False),
        Column("sales", Float, nullable=False),
        extend_existing=True,
    )

    metadata_obj.create_all(engine)

    stock_observations = [
        [1, 'ABC', 200, datetime(2023,1,1)], 
        [2, 'ABC', 202, datetime(2023,1,2)], 
        [3, 'ABC', 210, datetime(2023,1,3)],
        [4, 'ABC', 200, datetime(2023,1,4)], 
        [5, 'ABC', 202, datetime(2023,1,5)], 
        [6, 'XYZ', 210, datetime(2023,1,1)],
        [7, 'XYZ', 200, datetime(2023,1,2)], 
        [8, 'XYZ', 202, datetime(2023,1,3)], 
        [9, 'XYZ', 210, datetime(2023,1,4)],
        [10, 'XYZ', 200, datetime(2023,1,5)],  
    ]

    company_observations = [
        [1, 'ABC Corp', 'ABC'],
        [2, 'XYZ Corp', 'XYZ']
    ]

    sales_observations = [
        [1, 1, datetime(2023,1,1), 100],
        [2, 1, datetime(2023,1,2), 105],
        [3, 1, datetime(2023,1,3), 90],
        [4, 1, datetime(2023,1,4), 99],
        [5, 1, datetime(2023,1,5), 100],
        [6, 2, datetime(2023,1,1), 100],
        [7, 2, datetime(2023,1,2), 102],
        [8, 2, datetime(2023,1,3), 110],
        [9, 2, datetime(2023,1,4), 109],
        [10, 2, datetime(2023,1,5), 105],
    ]

    def insert_stock_obs(obs):
        stmt = insert(stocks).values(
            obs_id=obs[0],
            stock_ticker=obs[1],
            price=obs[2],
            date=obs[3]
        )
        with engine.begin() as conn:
            conn.execute(stmt)

    def insert_company_obs(obs):
        stmt = insert(company).values(
            company_id=obs[0],
            company_name=obs[1],
            stock_ticker=obs[2],
        )
        with engine.begin() as conn:
            conn.execute(stmt)

    def insert_sales_obs(obs):
        stmt = insert(sales).values(
            sales_id=obs[0],
            company_id=obs[1],
            date=obs[2],
            sales=obs[3]
        )
        with engine.begin() as conn:
            conn.execute(stmt)

    for obs in stock_observations:
        insert_stock_obs(obs)

    for obs in company_observations:
        insert_company_obs(obs)

    for obs in sales_observations:
        insert_sales_obs(obs)

    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "SQLQuery here if the user asked to generate query otherwise SQLResult"

    Only use the following tables:

    {table_info}

    Question: {input}"""
    PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )

    db = SQLDatabase(engine=engine)

    sql_chain = SQLDatabaseChain(llm=llm, 
                                    database=db, 
                                    prompt=PROMPT,
                                    verbose=verbose_flag, 
                                    use_query_checker=True)
    return sql_chain

def initialise_tools(llm, sql_chain):
    """
    Initialize the tools for the agent

    Parameters
    ----------
    llm : LLM object
    sql_chain : SQLDatabaseChain object

    Returns
    -------
    tools : Tool object
        The Tools set that is used by LLM to interact with the database
    """

    tools = load_tools(
        ["llm-math"],
        llm=llm
    )

    stock_tool = Tool(
        name='StockTable',
        func=sql_chain.run,
        description="Useful for when you need to answer questions about stocks and their prices."
    )

    company_tool = Tool(
        name='CompanyTable',
        func=sql_chain.run,
        description="Useful for when you need to answer questions about company."
    )
    
    sales_tool = Tool(
        name='SalesTable',
        func=sql_chain.run,
        description="Useful for when you need to answer questions about sales."
    )

    tools.append(stock_tool)
    tools.append(company_tool)
    tools.append(sales_tool)

    return tools


def initialise_zeroshot_agent(llm, tools, verbose_flag=False):
    """
    Initialize the zeroshot agent with the LLM and the SQL chain

    Parameters
    ----------
    llm : LLM object
    tools : Tool object
        The Tools set that is used by LLM to interact with the database
    verbose_flag : bool, optional
        The flag to indicate whether to print the output or not, by default False

    Returns
    -------
    Agent object
    """
    agent = initialize_agent(
        agent="zero-shot-react-description",
        llm=llm,
        tools=tools,
        verbose=verbose_flag,
        max_iterations=3
    )

    return agent

def initialise_conversational_agent(llm, tools, verbose_flag=False):
    """
    Initialize the conversational agent with the LLM and the SQL chain

    Parameters
    ----------
    llm : LLM object
    tools : Tool object
        The Tools set that is used by LLM to interact with the database
    verbose_flag : bool, optional
        The flag to indicate whether to print the output or not, by default False

    Returns
    -------
    Agent object
    """
    # Create a memory object to store the conversation history
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    agent = initialize_agent(
        agent="conversational-react-description",
        llm=llm,
        tools=tools,
        verbose=verbose_flag,
        max_iterations=3,
        memory=memory
    )

    return agent