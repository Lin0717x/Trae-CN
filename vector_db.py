import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

class VectorDBManager:
    def __init__(self, persist_directory="chroma_db", embedding_model="nomic-embed-text"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_db = None
        
    def init_db(self):
        """初始化或加载向量数据库"""
        try:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_documents"
            )
            print(f"向量数据库初始化成功，当前包含 {self.get_document_count()} 个文档")
        except Exception as e:
            print(f"初始化向量数据库时出错: {e}")
            self.vector_db = None
    
    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        """使用RecursiveCharacterTextSplitter分块文本"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def add_documents(self, documents):
        """向向量数据库添加文档"""
        if not self.vector_db:
            print("向量数据库未初始化")
            return
        
        all_docs = []
        for doc in documents:
            chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                metadata = {
                    "filename": doc["filename"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                all_docs.append(Document(page_content=chunk, metadata=metadata))
        
        if all_docs:
            self.vector_db.add_documents(all_docs)
            self.vector_db.persist()
            print(f"成功添加 {len(all_docs)} 个文本块")
        else:
            print("没有可添加的文档")
    
    def search(self, query, k=3):
        """检索与查询最相关的k个文本块"""
        if not self.vector_db:
            print("向量数据库未初始化")
            return []
        
        try:
            retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
            results = retriever.invoke(query)
            return results
        except Exception as e:
            print(f"检索时出错: {e}")
            return []
    
    def get_document_count(self):
        """获取向量数据库中文档数量"""
        if not self.vector_db:
            return 0
        try:
            return self.vector_db._collection.count()
        except Exception as e:
            print(f"获取文档数量时出错: {e}")
            return 0
    
    def clear_db(self):
        """清空向量数据库"""
        if not self.vector_db:
            print("向量数据库未初始化")
            return
        
        try:
            self.vector_db.delete_collection()
            self.vector_db = None
            self.init_db()
            print("向量数据库已清空")
        except Exception as e:
            print(f"清空向量数据库时出错: {e}")

if __name__ == "__main__":
    # 测试向量数据库管理器
    db_manager = VectorDBManager()
    db_manager.init_db()
    
    # 示例文档
    sample_docs = [
        {
            "filename": "sample1.txt",
            "content": "自然语言处理（NLP）是人工智能的一个重要分支，它使计算机能够理解、解释和生成人类语言。NLP技术在机器翻译、情感分析、文本摘要等领域有广泛应用。"
        },
        {
            "filename": "sample2.txt", 
            "content": "Transformer模型是Google在2017年提出的一种深度学习架构，它基于自注意力机制，在许多NLP任务上取得了突破性进展。BERT、GPT等著名模型都基于Transformer架构。"
        }
    ]
    
    db_manager.add_documents(sample_docs)
    
    # 测试检索
    results = db_manager.search("Transformer模型")
    print("\n检索结果:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result.page_content[:100]}...")
        print(f"   来源: {result.metadata['filename']}")