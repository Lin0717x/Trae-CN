#!/usr/bin/env python3
"""
RAG问答系统 - 命令行版本
"""

import os
import sys
from document_reader import batch_read_documents
from vector_db import VectorDBManager
from rag_qa import RAGQAChain

def main():
    print("=" * 60)
    print("RAG问答系统 - 命令行版本")
    print("=" * 60)
    
    # 初始化向量数据库
    db_manager = VectorDBManager()
    db_manager.init_db()
    
    # 检查是否有文档需要处理
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        docs = batch_read_documents(docs_folder)
        if docs:
            print(f"\n检测到 {len(docs)} 份文档，正在处理...")
            db_manager.add_documents(docs)
        else:
            print("\n未检测到文档，知识库为空")
    else:
        print(f"\n文档文件夹 {docs_folder} 不存在")
    
    # 创建问答链
    print("\n初始化问答链...")
    try:
        qa_chain = RAGQAChain(db_manager.vector_db.as_retriever())
        print("问答链初始化成功")
    except Exception as e:
        print(f"初始化问答链失败: {e}")
        print("请确保Ollama已安装并运行")
        sys.exit(1)
    
    # 开始问答循环
    print("\n" + "=" * 60)
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'clear' 清除对话历史")
    print("输入 'info' 查看知识库状态")
    print("=" * 60)
    
    while True:
        question = input("\n请输入问题: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ["quit", "exit"]:
            print("退出程序...")
            break
        
        if question.lower() == "clear":
            qa_chain.clear_history()
            print("对话历史已清除")
            continue
        
        if question.lower() == "info":
            doc_count = db_manager.get_document_count()
            print(f"知识库中文本块数量: {doc_count}")
            continue
        
        print("思考中...")
        answer, sources = qa_chain.ask(question)
        
        print(f"\n回答: {answer}")
        if sources:
            print("参考来源:")
            for i, source in enumerate(sources):
                print(f"  {i+1}. {source.metadata['filename']}")

if __name__ == "__main__":
    main()