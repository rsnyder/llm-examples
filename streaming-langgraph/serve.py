#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a simple FastAPI server that uses the LangGraph to chat with an OpenAI model.
It is a basic streaming example.  It does not include any chat history.
'''

import json
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, FileResponse

LLM_MODEL = 'gpt-4o'

llm = ChatOpenAI(model=LLM_MODEL)

class State(TypedDict):
    messages: list

def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)
graph = graph_builder.compile()

async def generate_chat_events(messages):
  for msg, metadata in graph.stream({'messages': messages}, stream_mode='messages'):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata['langgraph_node'] == 'chatbot'
    ):
        yield msg.content

app = FastAPI()

@app.get('/')
async def root():
  return FileResponse('../index.html')

@app.get('/chat/{prompt}')
@app.post('/chat')
async def chat(request: Request, prompt: Optional[str] = None, stream: Optional[bool] = False): 
  
  if request.method == 'POST':
    body = await request.body()
    payload = json.loads(body)
    prompt = payload.get('prompt', '')
    stream = payload.get('stream', False)

  messages = [
    ('system', 'You are a helpful assistant.'),
    ('user', prompt)
  ]

  if stream:
    return StreamingResponse(generate_chat_events(messages), media_type='text/event-stream')
  else:
    return Response(content=graph.invoke({'messages': messages})['messages'][-1].content , media_type='text/plain')  

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)