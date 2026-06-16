# RAG问答系统

## 项目简介

基于Ollama和LangChain构建的本地RAG（检索增强生成）问答系统，支持PDF和DOCX文档的上传、解析、向量化存储，并提供Web界面进行问答交互。

## 环境要求与安装步骤

### 系统要求
- Windows 10/11
- Python 3.8+
- Ollama

### 安装步骤

1. **安装Ollama**
   - 下载地址：https://ollama.com/download
   - 安装后运行：`ollama pull qwen2:7b` 和 `ollama pull nomic-embed-text`

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 模型下载
```bash
# 下载问答模型
ollama pull qwen2:7b

# 下载嵌入模型
ollama pull nomic-embed-text
```

## 使用说明

### 运行Web应用
```bash
streamlit run app.py
```

### 使用步骤
1. 在侧边栏上传PDF或DOCX文档
2. 点击"构建知识库"按钮
3. 在问答区输入问题并点击"提问"
4. 查看回答和对话历史

### 命令行版本
```bash
python cli_rag.py
```

## 关键技术点

### RAG流程
1. **文档加载**：支持PDF和DOCX格式的文档读取
2. **文本分块**：使用RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200
3. **向量化**：使用Ollama的nomic-embed-text嵌入模型


4. **向量存储**：采用Chroma向量数据库进行高效检索
5. **问答生成**：通过ConversationalRetrievalChain将检索结果与语言模型结合

### 核心技术栈
- **框架**: LangChain、Streamlit
- **语言模型**: Ollama (qwen2:7b)
- **嵌入模型**: Ollama (nomic-embed-text)
- **向量数据库**: Chroma

## 项目结构

```
RAG-QA-System/
├── app.py                 # Streamlit Web应用
├── cli_rag.py             # 命令行版本
├── document_reader.py     # 文档读取模块
├── vector_db.py           # 向量数据库管理
├── rag_qa.py              # RAG问答链
├── test_ollama.py         # 测试脚本
├── requirements.txt       # 依赖列表
├── docs/                  # 文档文件夹
└── chroma_db/             # 向量数据库存储
```

## 功能特点

- 支持PDF和DOCX文档上传
- 文档批量处理和向量化
- 对话历史记忆
- 基于文档的准确问答
- 对无关问题的拒答机制
- 本地部署，无需联网

## 已知问题与改进方向

### 已知问题
- 文档解析对复杂格式支持有限
- 大文档处理时间较长

### 改进方向
- 添加更多文档格式支持（如TXT、Markdown）
- 实现文档分块优化
- 添加文档删除功能
- 支持多语言问答

## 许可证

MIT License