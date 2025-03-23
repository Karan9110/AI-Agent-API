import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [  
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    # Parse the output string into a dictionary
    raw_output = raw_response.get("output")
    if isinstance(raw_output, str):
        # Remove any surrounding whitespace and parse the JSON string
        raw_output = json.loads(raw_output.strip())
    
    # Use PydanticOutputParser to parse the structured response
    structured_response = parser.parse_obj(raw_output.get("output")[0]["text"])
    print(structured_response)
except json.JSONDecodeError as e:
    print("Error decoding JSON:", e, "Raw Output -", raw_output)
except Exception as e:
    print("Error parsing response:", e, "Raw Response -", raw_response)