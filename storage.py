import os
from pathlib import Path

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

import faiss

from RAG.VectorBase import BaseVectorBase, WeaviateVectorBase
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat
from RAG.Embeddings import ZhipuEmbedding, OpenAIEmbedding
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # OpenAIEmbeddings,
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv('WEAVIATE_URL')

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPLATE="""
You are a useful assistant for provide professional customer question answering. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Must reply using the language used in the user’s question.
Context: {context}
Question: {question}
Answer:""",
    RAG_PROMPT_TEMPLATE_2="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。"""
)


# 1. 需要实现用户选择classname
# 2. 需要实现找到本地classes

def load_vectorstore(classname="example", filepath="", embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)):
    db_path = r'./faiss/'  # 用r""？
    path = Path(db_path)
    index_path = str(path / f"{classname}.faiss")  # output: faiss\example.faiss

    if len(filepath):
        print(f"Uploading {filepath} to the vector store.")
        # ---------- 上传数据---------- #
        # splitter = 'RecursiveCharacterTextSplitter'
        splitter = 'MarkdownTextSplitter'
        docs = ReadFiles(filepath, splitter).get_content(chunk_size=256, chunk_overlap=32)
        # print(docs)

        # ---------- 创建向量库---------- #
        print(f"Establishing the vector store.")
        # storage = WeaviateVectorBase(classname, docs)
        db = FAISS.from_documents(docs, embeddings)
        if os.path.exists(index_path):
            print(f"Class exists. Loading the local class.")
            # 如果存在，加载本地的向量库
            local_db = FAISS.load_local(folder_path=db_path, index_name=classname, embeddings=embeddings,
                                        allow_dangerous_deserialization=True)

            # 合并新的向量库到本地的向量库
            local_db.merge_from(db)

            # 保存合并后的向量库
            local_db.save_local(folder_path=db_path, index_name=classname)
        else:
            print(f"Creating new class \"{classname}\".")
            # 如果不存在，保存新的向量库
            db.save_local(folder_path=db_path, index_name=classname)
    else:
        # ---------- 获取向量库---------- #
        # storage = WeaviateVectorBase(classname)
        if os.path.exists(index_path):
            print(f"Class exists. Loading the local class.")
            db = FAISS.load_local(folder_path=db_path, index_name=classname, embeddings=embeddings,
                                  allow_dangerous_deserialization=True)
        else:
            print(f"Class does not exist. Please create or rename it.")
    # vectorstore = storage.get_vector(EmbeddingModel=OpenAIEmbedding())  # = ZhipuEmbedding()  # 创建EmbeddingModel

    return db


def get_query(classname, query, template=PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE'], embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)):
    # 读取已有向量库进行问答
    vectorstore = load_vectorstore(classname, embeddings=embeddings)
    print(vectorstore)
    # ---------- 1. 检索 ---------- #
    # 一旦向量数据库准备好，你就可以将它设定为检索组件，这个组件能够根据用户查询与已嵌入的文本块之间的语义相似度，来检索出额外的上下文信息。
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
    # (search_kwargs={"k": 3})  #
    # Q: as_retriever是什么东西？可以替换为什么？可以替换为query函数吗？
    # ---------- 2. 增强 ---------- #
    prompt = ChatPromptTemplate.from_template(template)
    # print(prompt)
    # ---------- 3. 生成 ---------- #
    llm = ChatOpenAI(model_name='gpt-4-0125-preview', api_key=OPENAI_API_KEY)
    """
        # RetrievalQAWithSourcesChain
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        return chain({"question": query}, return_only_outputs=True)
    """
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    #  ---------- 4. 回答 ---------- #
    print(f"Searching for relevant documents...")
    docs = retriever.get_relevant_documents(query)  # 与.similarity_search没有区别
    text_ls = [doc.page_content[:] for doc in docs]
    ans = rag_chain.invoke(query)
    return ans, text_ls


# vector.persist()  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector = VectorStore()
# vector.load_vector('./storage') # 加载本地的数据库
# embedding = ZhipuEmbedding() # 创建EmbeddingModel

# question = '请解释合同的定义。'
# content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
# print("检索到的知识：" + content)
# chat = OpenAIChat(model='gpt-4-0125-preview')
# print("回答：" + chat.chat(question, [], content))
