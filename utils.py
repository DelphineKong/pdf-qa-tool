from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Tongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.vectorstores import chroma
from langchain_community.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain

import os
def qa_agent(api_key,memory,uploaded_file,question):
    # 设置 API-KEY
    os.environ["DASHSCOPE_API_KEY"] = api_key
    # 使用 Tongyi LLM
    llm = Tongyi()
    # 将用户上传的文件写入本地的pdf文件，并load获取
    file_content=uploaded_file.read()
    temp_file_path="temp.pdf"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)
    loader=PyPDFLoader(temp_file_path)
    docs=loader.load()
    # 分割
    texts_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts=texts_splitter.split_documents(docs)
    # 阿里嵌入模型
    embeddings_model=DashScopeEmbeddings()
    # 将文件数据向量化并导入数据库
    db=faiss.FAISS.from_documents(texts,embeddings_model)
    # chroma.Chroma.from_documents()
    # 建一个检索器
    retriever=db.as_retriever()
    # 创建链
    qa=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    response=qa.invoke({"chat_history":memory,"question":question})
    return response


# print(get_chat_response("牛顿提出过哪些知名的定律？",memory,os.getenv("DASHSCOPE_API_KEY")))
# print(get_chat_response("我上一个问题是什么？",memory,os.getenv("DASHSCOPE_API_KEY")))

