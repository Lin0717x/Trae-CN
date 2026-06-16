import streamlit as st
import os
import tempfile
from document_reader import read_document
from vector_db import VectorDBManager
from rag_qa import RAGQAChain

# 设置页面配置
st.set_page_config(
    page_title="RAG问答系统",
    page_icon="📚",
    layout="wide"
)

# 初始化会话状态
if "db_manager" not in st.session_state:
    st.session_state.db_manager = VectorDBManager()
    st.session_state.db_manager.init_db()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

def init_qa_chain():
    """初始化问答链"""
    try:
        st.session_state.qa_chain = RAGQAChain(
            st.session_state.db_manager.vector_db.as_retriever()
        )
        return True
    except Exception as e:
        st.error(f"初始化问答链失败: {e}")
        return False

def process_uploaded_files(files):
    """处理上传的文件"""
    documents = []
    for file in files:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name
        
        # 读取文档
        text = read_document(temp_path)
        if text:
            documents.append({
                "filename": file.name,
                "content": text
            })
        
        # 删除临时文件
        os.unlink(temp_path)
    
    return documents

# 页面标题
st.title("📚 RAG问答系统")

# 侧边栏
with st.sidebar:
    st.header("知识库管理")
    
    # 文档上传
    uploaded_files = st.file_uploader(
        "上传PDF或DOCX文件",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    # 构建知识库按钮
    if st.button("📥 构建知识库"):
        if uploaded_files:
            with st.spinner("正在处理文档..."):
                docs = process_uploaded_files(uploaded_files)
                if docs:
                    st.session_state.db_manager.add_documents(docs)
                    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
                    st.success(f"成功处理 {len(docs)} 份文档")
                    # 重新初始化问答链
                    init_qa_chain()
                else:
                    st.error("未能处理任何文档")
        else:
            st.warning("请先上传文档")
    
    # 清空知识库
    if st.button("🗑️ 清空知识库"):
        st.session_state.db_manager.clear_db()
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.session_state.qa_chain = None
        st.success("知识库已清空")
    
    # 知识库状态
    st.divider()
    st.subheader("知识库状态")
    doc_count = st.session_state.db_manager.get_document_count()
    st.info(f"文本块数量: {doc_count}")
    if st.session_state.uploaded_files:
        st.write("已上传文档:")
        for filename in st.session_state.uploaded_files:
            st.write(f"- {filename}")

# 主内容区
col1, col2 = st.columns([3, 1])

with col1:
    st.header("问答交互")
    
    # 对话历史
    if st.session_state.chat_history:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
    
    # 问题输入
    question = st.text_input("请输入您的问题:", key="question_input")
    
    if st.button("提问", type="primary"):
        if question.strip():
            if not st.session_state.qa_chain:
                if not init_qa_chain():
                    st.error("请先构建知识库")
                else:
                    st.warning("已自动初始化问答链")
            
            if st.session_state.qa_chain:
                with st.spinner("思考中..."):
                    answer, sources = st.session_state.qa_chain.ask(question)
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
            else:
                st.error("无法回答，请先构建知识库")
        else:
            st.warning("请输入问题")

with col2:
    st.header("操作提示")
    st.markdown("""
    **使用步骤:**
    
    1. 在侧边栏上传PDF或DOCX文档
    2. 点击"构建知识库"按钮
    3. 在问答区输入问题并点击"提问"
    4. 查看回答和对话历史
    
    **注意事项:**
    
    - 支持PDF和DOCX格式
    - 系统会记住对话上下文
    - 如果文档中没有相关信息，会显示"文档中未找到相关答案"
    """)
    
    st.divider()
    st.subheader("快捷键")
    st.markdown("""
    - Enter: 提交问题
    """)

# 页脚
st.divider()
st.caption("RAG问答系统 - 基于Ollama和LangChain构建")