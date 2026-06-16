import os
from PyPDF2 import PdfReader
from docx import Document

def read_pdf(file_path):
    """读取PDF文件并提取文本"""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"读取PDF文件 {file_path} 时出错: {e}")
        return ""

def read_docx(file_path):
    """读取DOCX文件并提取文本"""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"读取DOCX文件 {file_path} 时出错: {e}")
        return ""

def read_document(file_path):
    """根据文件扩展名读取文档"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        print(f"不支持的文件格式: {ext}")
        return ""

def batch_read_documents(folder_path):
    """批量读取文件夹内所有文档"""
    documents = []
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return documents
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in [".pdf", ".docx"]:
                print(f"正在读取: {filename}")
                text = read_document(file_path)
                if text:
                    documents.append({
                        "filename": filename,
                        "content": text
                    })
    return documents

if __name__ == "__main__":
    docs = batch_read_documents("docs")
    print(f"共读取 {len(docs)} 份文档")
    for doc in docs:
        print(f"- {doc['filename']}: {len(doc['content'])} 字符")