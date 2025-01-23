#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
QA Chat Server
'''

import json, secrets
from typing import Optional

# Define LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini')

# Define embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Define vector store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# Get content and chunk it
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=('post-content', 'post-title', 'post-header')
        )
    )
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
vector_store.add_documents(documents=all_splits)

# Create graph

from langgraph.graph import MessagesState, StateGraph
graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
 
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage
async def generate_chat_events(messages, config):
  for msg, metadata in graph.stream({'messages': messages}, config, stream_mode='messages'):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata['langgraph_node'] == 'query_or_respond' or metadata['langgraph_node'] == 'generate' 
    ):
        yield msg.content

 
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, FileResponse

app = FastAPI()

@app.get('/')
async def root():
  return FileResponse('../../index.html')

@app.get('/chat/{prompt}')
@app.post('/chat')
async def chat(request: Request, prompt: Optional[str] = None, sessionid: Optional[str] = None, stream: Optional[bool] = False): 
  
  if request.method == 'POST':
    body = await request.body()
    payload = json.loads(body)
    prompt = payload.get('prompt', '')
    stream = payload.get('stream', False)
    sessionid = payload.get('sessionid', secrets.token_hex(4))

  config = {'configurable': {'thread_id': sessionid}}
  messages = [{'type': 'user', 'content': prompt}]
  
  if stream:
    return StreamingResponse(generate_chat_events([{'type': 'user', 'content': prompt}], config), media_type='text/event-stream')
  else:
    return Response(content=graph.invoke({'messages': messages}, config)['messages'][-1].content , media_type='text/plain')  

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)