import streamlit as stl
import langchain_helper as lch
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

stl.title("YouTube Assistant")

if 'db' not in stl.session_state:
    stl.session_state["db"] = None
if 'processed_url' not in stl.session_state:
    stl.session_state["processed_url"] = ""
if 'memory' not in stl.session_state:
    stl.session_state["memory"] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
if 'rag_chain' not in stl.session_state:
    stl.session_state['rag_chain'] = None

with stl.sidebar:
    stl.header("1. Load Video")
    url = stl.text_input("Enter YouTube Video URL")

    if stl.button("Process Video"):
        if url:
            if stl.session_state["processed_url"] != url:
                with stl.spinner("Building Vector Database..."):
                    try:
                        stl.session_state["db"] = lch.create_vector_db(url)
                        stl.session_state["memory"] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
                        stl.session_state['rag_chain'] = lch.get_rag_chain_with_memory(stl.session_state["db"])
                        stl.session_state["processed_url"] = url
                        stl.success("Video loaded! You can now ask questions about the video.")
                       
                    except Exception as e:
                        stl.error(f"Error processing video: {e}")
                        stl.session_state["db"] = None


            else:
                stl.info("This video has already been processed.")
        
        else:
            stl.warning("Please enter a valid YouTube URL.")

if stl.session_state["db"]:
    stl.subheader(f"Chatting about: {stl.session_state['processed_url']}")

    if prompt := stl.chat_input("Ask me about the video...."):
        history = stl.session_state["memory"].load_memory_variables({})["chat_history"]
        if isinstance(history, list):
            for msg in history:
                if msg.type == "human":
                    with stl.chat_message("user"):
                        stl.markdown(msg.content)
                elif msg.type == "ai":
                    with stl.chat_message("assistant"):
                        stl.markdown(msg.content)
        with stl.chat_message("user"):
            stl.markdown(prompt)
        with stl.chat_message("assistant"):
            with stl.spinner("Generating response..."):
                response = stl.session_state['rag_chain'].invoke({
                    "question": prompt,
                    "chat_history":stl.session_state["memory"].load_memory_variables({})["chat_history"]
                })
                stl.markdown(response)
                stl.session_state["memory"].save_context({"input": prompt}, {"output": response})

else:
    stl.info("Please load a YouTube video to start chatting.")
