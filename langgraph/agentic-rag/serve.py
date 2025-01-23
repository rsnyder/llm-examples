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

# Create graph

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
 
from langchain_core.messages import HumanMessage
async def generate_chat_events(messages, config):
  for msg, metadata in agent_executor.stream({'messages': messages}, config, stream_mode='messages'):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata['langgraph_node'] == 'agent' 
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