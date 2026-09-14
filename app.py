import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# إعدادات الواجهة المؤسسية
st.set_page_config(
    page_title="Enterprise Space AI Assistant", layout="wide"
)
st.title("🚀 Enterprise AI Assistant for Space Documentation")
st.markdown(
    "مساعد ذكي ومؤمن للبحث وتحليل الوثائق الهندسية والفضائية للشركات."
)

# التحقق من مفتاح الـ API الآمن من إعدادات السيرفر (Secrets)
if "GEMINI_API_KEY" in st.secrets:
  os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
  st.error(
      "خطأ أمني: مفتاح GEMINI_API_KEY غير مُعد في إعدادات المنصة (Secrets)."
  )
  st.stop()

DATA_DIR = "./data"
DB_DIR = "./vector_db"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

# رفع ملفات الـ PDF
uploaded_file = st.file_uploader(
    "اختر ملف PDF تقني أو فضائي لرفعه", type=["pdf"]
)
if uploaded_file is not None:
  file_path = os.path.join(DATA_DIR, uploaded_file.name)
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
  st.success(f"تم رفع الملف بنجاح: {uploaded_file.name}")

# زر معالجة وفهرسة المستندات
if st.button("معالجة المستندات وفهرستها"):
  with st.spinner("جاري قراءة وتقسيم المستندات وبناء المتجهات..."):
    if not os.listdir(DATA_DIR):
      st.warning("الرجاء رفع ملفات PDF أولاً.")
    else:
      loader = DirectoryLoader(DATA_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)
      docs = loader.load()

      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000, chunk_overlap=200
      )
      splits = text_splitter.split_documents(docs)

      embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
      vectorstore = Chroma.from_documents(
          documents=splits, embedding=embeddings, persist_directory=DB_DIR
      )
      st.success("تم الانتهاء من فهرسة المستندات بنجاح!")

# قسم محادثة المستخدم والسؤال والجواب
st.subheader("اسأل المستندات التقنية")
query = st.text_input(
    "اكتب سؤالك هنا (مثال: What are the telemetry requirements؟)"
)

if query:
  if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    with st.spinner("جاري البحث وتوليد الإجابة الحية..."):
      embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
      vectorstore = Chroma(
          persist_directory=DB_DIR, embedding_function=embeddings
      )

      # استرجاع أقرب المقاطع المطابقة للسؤال
      retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
      relevant_docs = retriever.invoke(query)

      # دمج النصوص المسترجعة في سياق موحد
      context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

      # تجهيز البرومبت وإرساله لنموذج Gemini مباشرة
      llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

      prompt = f"""
أنت مساعد هندسي ذكي ومحترف لتحليل وثائق الفضاء والطيران.
اعتمد حصرياً على السياق التقني المسترجع أدناه للإجابة على السؤال بدقة عالية.
إذا لم تكن الإجابة موجودة في السياق، قل "لا توجد معلومات كافية في المستندات للإجابة".

السياق:
{context_text}

السؤال:
{query}

الإجابة الاحترافية:
"""

      response = llm.invoke(prompt)

      st.markdown("### الإجابة:")
      st.write(response.content)

      with st.expander("عرض المقاطع المستخدمة من المستندات (Context)"):
        for i, doc in enumerate(relevant_docs):
          st.markdown(f"**المقطع {i+1}:**")
          st.write(doc.page_content)
  else:
    st.warning("الرجاء معالجة المستندات وفهرستها أولاً قبل البدء بالسؤال.")
