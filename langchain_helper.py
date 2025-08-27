from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import re


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def create_vector_db(video_url):
    """Create vector database from YouTube video transcript"""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    try:
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)
        full_text = " ".join([snippet.text for snippet in transcript_data.snippets])

        document = Document(page_content=full_text, metadata={"source": video_url})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents([document])
        embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db
        
    except Exception as e:
        print(f"Error loading transcript: {e}")
        raise


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_rag_chain_with_memory(db: FAISS) -> RunnablePassthrough:
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    prompt_template = PromptTemplate(
        input_variables=["chat_history","context","question"],
        template="""
        You are a helpful YouTube assistant that can answer questions about the video based on the provided transcript.
        Current conversation:
        {chat_history}
        Answer the following question: {question}
        By using the following context from the video transcript:
        {context}
        Answer the user's question with the help of the provided video transcript and chat history only.
        If the answer is not contained in the provided context, say "The video does not have information on this topic."
        Do not make up any information. Your answers should be detailed.
        """)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

