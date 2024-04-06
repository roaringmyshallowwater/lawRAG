#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str, splitter: str = 'RecursiveCharacterTextSplitter') -> None:
        self._path = path
        self.file_list = self.get_files()
        self.splitter = splitter

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, chunk_size=512, chunk_overlap=64):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs.extend(chunk_content)
        return docs

    def get_chunk(self, doc: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        if self.splitter == 'RecursiveCharacterTextSplitter':
            return self.chunk_recursive_character(doc, chunk_size, chunk_overlap)
        elif self.splitter == 'MarkdownTextSplitter':
            return self.chunk_markdown(doc, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported splitter: {self.splitter}")

    def chunk_recursive_character(self, doc: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        model_path = os.environ.get("BAICHUAN_PATH", "baichuan-inc/Baichuan2-13B-Chat")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        length_function = lambda text: len(tokenizer.tokenize(text))

        splitter = RecursiveCharacterTextSplitter(
            separators=["，", "。", '\\n\\n', '\\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )
        return splitter.create_documents([doc])

    def chunk_markdown(self, markdown_text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = markdown_splitter.create_documents([markdown_text])
        # print(docs[0].page_content[:])
        # text_chunks = [doc.page_content[:] for doc in docs]
        return docs
    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        loader = UnstructuredPDFLoader(file_path)
        docs = loader.load()
        return docs[0].page_content[:]

    @classmethod
    def read_markdown(cls, file_path: str):
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        return docs[0].page_content[:]

    @classmethod
    def read_text(cls, file_path: str):
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        return docs[0].page_content[:]


class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
