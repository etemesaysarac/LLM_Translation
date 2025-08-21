import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- sadece görsel sunum için ---
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

# Model
llm = ChatOpenAI(model="gpt-4o-mini")  # istersen yine gpt-3.5-turbo yazabilirsin

# Kaynak sayfayı yükle
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Böl ve vektörleştir
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Yardımcı
def format_docs(docs_):
    return "\n\n".join(doc.page_content for doc in docs_)

# ---- MARKDOWN ODAKLI RAG PROMPT ----
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior technical writer. Write a clear, accurate article in "
                "GitHub‑flavored Markdown using ONLY the supplied context. "
                "Structure strictly as:\n\n"
                "# {auto_title}\n"
                "## TL;DR\n- 3–5 bullets\n\n"
                "## Key Points\n- bullets with brief explanations\n\n"
                "## Detailed Explanation\n### Subtopics as needed with short paragraphs\n"
                "Include mini examples if relevant.\n\n"
                "## References\n- list the source links (if present in context)\n\n"
                "If information is missing, say so. Keep it concise and well formatted."
            ),
        ),
        (
            "user",
            "Question: {question}\n\nContext:\n{context}\n"
            "Generate a good `auto_title` for the article."
        ),
    ]
)

# RAG zinciri
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        # küçük bir hile: başlığa soru kısa şekli verilebilir
        "auto_title": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "What is Task Decomposition?"
    md_text = rag_chain.invoke(question)   # tek parça çıktı (markdown)
    # Konsolda şık gösterim
    console = Console()
    console.print(Markdown(md_text))
