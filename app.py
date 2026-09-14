import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# إعدادات الواجهة
st.set_page_config(page_title="Local AI Assistant for Space Documentation", layout="wide")
st.title("🚀 AI Assistant for Space Documentation")
st.markdown("مساعد ذكي للبحث وتحليل الوثائق الهندسية والفضائية باستخدام RAG و Google Gemini.")

# إدخال مفتاح الـ API عبر الواجهة أو استخدام المتغيرات البيئية
api_key = st.text_input("أدخل Google Gemini API Key الخاص بك:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    DATA_DIR = "./data"
    DB_DIR = "./vector_db"

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # رفع ملفات الـ PDF
    uploaded_file = st.file_uploader("اختر ملف PDF تقني أو فضائي لرفعه", type=["pdf"])
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

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # استخدام Google Gemini Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=DB_DIR
                )
                st.success("تم الانتهاء من فهرسة المستندات بنجاح!")

    # قسم محادثة المستخدم والسؤال والجواب
    st.subheader("اسأل المستندات التقنية")
    query = st.text_input("اكتب سؤالك هنا (مثال: What are the telemetry requirements؟)")

    if query:
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            with st.spinner("جاري البحث وتوليد الإجابة..."):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # استخدام نموذج Gemini للتوليد
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )

                response = qa_chain.invoke({"query": query})

                st.markdown("### الإجابة:")
                st.write(response["result"])

                with st.expander("عرض المقاطع المستخدمة من المستندات (Context)"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**المقطع {i + 1}:**")
                        st.write(doc.page_content)
        else:
            st.warning("الرجاء معالجة المستندات وفهرستها أولاً قبل البدء بالسؤال.")
else:
    st.info("الرجاء إدخال مفتاح الـ API الخاص بـ Google Gemini للبدء في تشغيل التطبيق.")