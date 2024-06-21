import os
import subprocess
import getpass
import operator
import functools
from typing import Annotated, Sequence, TypedDict
from flask import Flask, request, render_template, jsonify
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.document_loaders import PyMuPDFLoader, BSHTMLLoader, CSVLoader, TextLoader, JSONLoader

# Function to uninstall matplotlib if it's installed
def uninstall_matplotlib():
    try:
        subprocess.check_call(["python", "-c", "import matplotlib"])
        print("Uninstalling matplotlib...")
        subprocess.check_call(["pip", "uninstall", "-y", "matplotlib"])
        print("Matplotlib has been successfully uninstalled.")
    except subprocess.CalledProcessError:
        print("Matplotlib is not installed.")

# Uninstall matplotlib if it's present
uninstall_matplotlib()

# Setting API keys and environment variables
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("sk-proj-PAZDiW59Dm9D4QEBU54HT3BlbkFJpBFYnu2vjJzXmKTLYAIN")
_set_if_undefined("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")

TAVILY_API_KEY = "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"
tavilySearchAPIWrapper = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
tavily_tool = TavilySearchResults(api_wrapper=tavilySearchAPIWrapper, max_results=5)

python_repl_tool = PythonREPLTool()

# Initialize Flask app
app = Flask(__name__)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Coder", "Mentor"]

system_prompt = ( # prompt engineer to ensure that Coder handles all coding tasks
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. If task is coding-related,"
    " designate to Coder first. For anything else, send to Mentor first."
    " Each worker will perform a task and respond with their results"
    " and status. When finished, respond with FINISH."
)

options = ["FINISH"] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-1106-preview",
                 openai_api_key="sk-proj-PAZDiW59Dm9D4QEBU54HT3BlbkFJpBFYnu2vjJzXmKTLYAIN")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data."
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

mentor_agent = create_agent(
    llm,
    [tavily_tool],
    "You provide guidance for interviews, classes, revision tasks, and general questions."
)
mentor_node = functools.partial(agent_node, agent=mentor_agent, name="Mentor")

workflow = StateGraph(AgentState)
workflow.add_node("Mentor", mentor_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")
graph = workflow.compile()

initial_state = AgentState(messages=[HumanMessage(content="Please input a greeting to start your interaction :)")], next="supervisor")
state = initial_state.copy()

def extract_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "File does not exist."
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".html":
        return extract_text_from_html(file_path)
    elif file_extension == ".txt":
        return extract_text_from_txt(file_path)
    elif file_extension == ".json":
        return extract_text_from_json(file_path)
    elif file_extension == ".csv":
        return extract_text_from_csv(file_path)
    else:
        return "Unsupported file type."

def concatenate_text(doc_data) -> str:
    text = ""
    for page in doc_data:
        text += page.page_content
        text += '\n'
    return text

def extract_text_from_pdf(file_path: str) -> str:
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    return concatenate_text(data)

def extract_text_from_html(file_path: str) -> str:
    loader = BSHTMLLoader(file_path)
    data = loader.load()
    return concatenate_text(data)

def extract_text_from_txt(file_path: str) -> str:
    loader = TextLoader(file_path)
    data = loader.load()
    return concatenate_text(data)

def extract_text_from_json(file_path: str) -> str:
    loader = JSONLoader(
        file_path,
        jq_schema='.messages[].content',
        text_content=False
    )
    data = loader.load()
    return concatenate_text(data)

def extract_text_from_csv(file_path: str) -> str:
    loader = CSVLoader(file_path)
    data = loader.load()
    return concatenate_text(data)

def serialize_state(state):
    return {
        "messages": [{"content": message.content} for message in state["messages"]],
        "next": state["next"],
        "question": state.get("question", 1),
        "content": state.get("content", "")
    }

def deserialize_state(state):
    return {
        "messages": [HumanMessage(content=message["content"]) for message in state.get("messages", [])],
        "next": state.get("next", "supervisor"),
        "question": state.get("question", 1),
        "content": state.get("content", "")
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global state
    data = request.json
    user_input = data.get('input')
    response = ""

    if 'state' in data:
        state = deserialize_state(data['state'])

    if 'question' not in state:
        state['question'] = 1

    if state['question'] == 1:
        state['question'] = 2
        response = "Does your task require a path or link? (Yes/No)"
    elif state['question'] == 2:
        if user_input.lower() == 'yes':
            state['question'] = 3
            response = "Please provide the path or link."
        else:
            state['question'] = 4
            state['content'] = ""
            response = "What can I help you with today? (enter exit/quit/stop to end conversation):"
    elif state['question'] == 3:
        file_path = user_input.strip()
        content = extract_text(file_path)
        state['content'] = content
        state['question'] = 4
        response = "What can I help you with today? (enter exit/quit/stop to end conversation):"
    elif state['question'] == 4:
        if user_input.lower() in ["exit", "quit", "stop"]:
            state = initial_state.copy()
            response = "Conversation ended."
        else:
            content = state.get('content', "")
            request_message = HumanMessage(content=f"{user_input}\n\n{content}")
            state["messages"].append(request_message)

            last_message = ""
            agent_name = "Supervisor"  # Default to Supervisor if not set
            for s in graph.stream(state, {"recursion_limit": 10}):
                if "__end__" not in s:
                    for key, value in s.items():
                        if isinstance(value, dict) and "messages" in value and value["messages"]:
                            last_message = value["messages"][-1].content
                            agent_name = key

            state["messages"].extend(s.get("messages", []))
            state["next"] = s.get("next", "supervisor")

            response = f"({agent_name} speaking): {last_message}"
            state['question'] = 2  # Go back to asking if the task requires a path or link
            response += "\n\nDoes your task require a path or link? (Yes/No)"

    return jsonify({"response": response, "state": serialize_state(state)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
