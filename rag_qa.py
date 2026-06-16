from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class RAGQAChain:
    def __init__(self, retriever, model_name="qwen2:7b"):
        self.model_name = model_name
        self.retriever = retriever
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """构建ConversationalRetrievalChain"""
        # 系统提示词
        system_prompt = """
你是一个基于文档的问答助手。请严格按照以下规则回答问题：

1. 必须基于提供的参考文档内容进行回答
2. 如果文档中有相关信息，用简洁明了的语言总结回答
3. 如果文档中没有相关信息，直接回答"文档中未找到相关答案"
4. 不要编造信息，不要添加文档中没有的内容
5. 如果有多个相关文档片段，综合所有信息进行回答

参考文档：
{context}

问题：
{question}

回答：
"""
        
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
        
        return chain
    
    def ask(self, question):
        """回答问题"""
        try:
            result = self.chain({"question": question})
            answer = result.get("answer", "").strip()
            sources = result.get("source_documents", [])
            
            # 检查是否需要拒绝回答
            if not sources or all(len(doc.page_content.strip()) == 0 for doc in sources):
                return "文档中未找到相关答案", []
            
            # 如果答案不是来自文档，也拒绝回答
            if "文档中未找到相关答案" not in answer and not self._is_answer_from_context(answer, sources):
                # 检查是否真的有相关上下文
                context_text = "\n".join([doc.page_content for doc in sources])
                if not self._has_relevant_context(question, context_text):
                    return "文档中未找到相关答案", []
            
            return answer, sources
        except Exception as e:
            print(f"问答链执行出错: {e}")
            return "文档中未找到相关答案", []
    
    def _is_answer_from_context(self, answer, sources):
        """检查答案是否来自上下文"""
        context_text = " ".join([doc.page_content for doc in sources]).lower()
        answer_text = answer.lower()
        
        # 检查答案中的关键词是否在上下文中出现
        answer_words = answer_text.split()
        context_words = context_text.split()
        
        # 如果答案中有超过50%的内容不在上下文中，可能是幻觉
        unknown_words = [word for word in answer_words if word not in context_words]
        if len(unknown_words) > len(answer_words) * 0.5:
            return False
        return True
    
    def _has_relevant_context(self, question, context):
        """检查是否有相关上下文"""
        question_words = question.lower().split()
        context_words = context.lower().split()
        
        # 检查问题中是否有至少2个关键词在上下文中出现
        matching_words = [word for word in question_words if word in context_words]
        return len(matching_words) >= 2
    
    def clear_history(self):
        """清除对话历史"""
        self.memory.clear()
    
    def get_history(self):
        """获取对话历史"""
        return self.memory.chat_memory.messages

if __name__ == "__main__":
    # 测试RAG问答链
    from vector_db import VectorDBManager
    
    # 初始化向量数据库
    db_manager = VectorDBManager()
    db_manager.init_db()
    
    # 测试问答链
    qa_chain = RAGQAChain(db_manager.vector_db.as_retriever())
    
    # 测试问题
    test_questions = [
        "什么是自然语言处理？",
        "Transformer模型是什么时候提出的？",
        "BERT模型基于什么架构？",
        "什么是机器学习？",
        "今天天气怎么样？"
    ]
    
    print("测试RAG问答链:\n")
    for question in test_questions:
        print(f"问题: {question}")
        answer, sources = qa_chain.ask(question)
        print(f"回答: {answer}")
        if sources:
            print(f"来源: {[s.metadata['filename'] for s in sources]}")
        print()
    
    qa_chain.clear_history()