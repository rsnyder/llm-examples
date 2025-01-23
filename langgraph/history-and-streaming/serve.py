#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a simple FastAPI server that uses the LangGraph to chat with an OpenAI model.
It demonstrates streaming with chat history.
'''

import json, secrets
from typing import Optional

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, FileResponse

LLM_MODEL = 'gpt-4o'

llm = ChatOpenAI(model=LLM_MODEL)
memory = MemorySaver()

def chatbot(state: MessagesState):
    return {'messages': [llm.invoke(state['messages'])]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile(checkpointer=memory)

async def generate_chat_events(messages, config):
  for msg, metadata in graph.stream({'messages': messages}, config, stream_mode='messages'):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata['langgraph_node'] == 'chatbot'
    ):
        yield msg.content

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