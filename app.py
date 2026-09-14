import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
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

      embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
      vectorstore = Chroma.from_documents(
          documents=splits, embedding=embeddings, persist_directory=DB_DIR
      )
      st.success("تم الانتهاء من فهرسة المستندات بنجاح!")

# قسم محادثة المستخدم والسؤال والجواب
st.subheader("اسأل المستندات التقنية")
query = st.text_input("اكتب سؤالك هنا (مثال: What are the telemetry requirements؟)")

if query:
  if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    with st.spinner("جاري البحث وتوليد الإجابة الحية..."):
      embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
      vectorstore = Chroma(
          persist_directory=DB_DIR, embedding_function=embeddings
      )
      retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

      # استخدام نموذج Gemini الحديث
      llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

      # بناء برومبت مخصص واحترافي للشركات
      system_prompt = (
          "أنت مساعد هندسي ذكي ومحترف لتحليل وثائق الفضاء والطيران."
          "اعتمد حصرياً على السياق المسترجع أدناه للإجابة على السؤال بدقة تقنية عالية."
          "إذا لمא تكن تعرف الإجابة، قل لا أعرف.\n\n"
          "السياق:\n{context}"
      )

      prompt = ChatPromptTemplate.from_messages([
          ("system", system_prompt),
          ("human", "{input}"),
      ])

      # إنشاء سلسلة الإجابة الحديثة (LCEL)
      question_answer_chain = create_stuff_documents_chain(llm, prompt)
      rag_chain = create_retrieval_chain(retriever, question_answer_chain)

      response = rag_chain.invoke({"input": query})

      st.markdown("### الإجابة:")
      st.write(response["answer"])

      with st.expander("عرض المقاطع المستخدمة من المستندات (Context)"):
        for i, doc in enumerate(response["context"]):
          st.markdown(f"**المقطع {i+1}:**")
          st.write(doc.page_content)
  else:
    st.warning("الرجاء معالجة المستندات وفهرستها أولاً قبل البدء بالسؤال.")
