#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Tuple, Union
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""
You are a useful assistant for provide professional customer question answering. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Must reply using the language used in the user’s question.
Context: {context}
Question: {question}
Answer:""",
    RAG_PROMPT_TEMPALTE_2="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。"""
)


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, query: str, history: List[dict], retriever) -> str:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'])
        llm = ChatOpenAI(model_name=self.model, temperature=0)
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        return rag_chain.invoke(query)

    def chat_2(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1
        )
        return response.choices[0].message.content

    def get_stream_response(self, prompt: str, content: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        history = [{'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)}]
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1,
            stream=True
        )

        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=128):
                if chunk:
                    yield chunk.decode('utf-8')
        else:
            raise Exception(f"Request failed with status code {response.status_code}")