#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Simple agent that uses Claude LLM and Tavily for web search
'''

import json, secrets
from typing import Optional

# Define LLM
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model='gpt-4o-mini')

from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")

# Define tools
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=2)
tools = [search]

# Define memory
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# Use LangGraph prebuilt ReAct agent constructor - https://langchain-ai.github.io/langgraph/how-tos/#langgraph.prebuilt.chat_agent_executor.create_react_agent
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(model, tools, checkpointer=memory)

from langchain_core.messages import HumanMessage
def get_response(messages, config):
    response = []
    for msg, metadata in agent_executor.stream({'messages': messages}, config, stream_mode='messages'):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata['langgraph_node'] == 'agent' 
        ):
            response.append(''.join([item['text'] for item in msg.content if 'text' in item]))
    return ''.join(response)

async def stream_response(messages, config):
  for msg, metadata in agent_executor.stream({'messages': messages}, config, stream_mode='messages'):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata['langgraph_node'] == 'agent' 
    ):
        yield ''.join([item['text'] for item in msg.content if 'text' in item])
 
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
  messages = [HumanMessage(content=prompt)]
  
  if stream:
    return StreamingResponse(stream_response(messages, config), media_type='text/event-stream')
  else:
    return Response(content=get_response(messages, config) , media_type='text/plain')  

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)