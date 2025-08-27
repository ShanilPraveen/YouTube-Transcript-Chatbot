import streamlit as stl
import langchain_helper as lch
from langchain_core.messages import HumanMessage, AIMessage

stl.title("YouTube Assistant")

if 'db' not in stl.session_state:
    stl.session_state["db"] = None
if 'processed_url' not in stl.session_state:
    stl.session_state["processed_url"] = ""
if 'messages' not in stl.session_state:
    stl.session_state["messages"] = []

with stl.sidebar:
    stl.header("1. Load Video")
    url = stl.text_input("Enter YouTube Video URL")

    if stl.button("Process Video"):
        if url:
            if stl.session_state["processed_url"] != url:
                with stl.spinner("Building Vector Database..."):
                    try:
                        stl.session_state["db"] = lch.create_vector_db(url)
                        stl.session_state['rag_chain'] = lch.get_rag_chain_with_memory(stl.session_state["db"])
                        stl.session_state["processed_url"] = url
                        stl.success("Video loaded! You can now ask questions about the video.")
                        stl.session_state["messages"] = []
                       
                    except Exception as e:
                        stl.error(f"Error processing video: {e}")
                        stl.session_state["db"] = None


            else:
                stl.info("This video has already been processed.")
        
        else:
            stl.warning("Please enter a valid YouTube URL.")

if stl.session_state["db"]:
    stl.subheader(f"Chatting about: {stl.session_state['processed_url']}")
    for message in stl.session_state["messages"]:
        with stl.chat_message(message["role"]):
            stl.markdown(message["content"])


    if prompt := stl.chat_input("Ask me about the video...."):
        stl.session_state["messages"].append({"role": "user", "content": prompt})
        with stl.chat_message("user"):
            stl.markdown(prompt)

        chat_history=[]
        for msg in stl.session_state["messages"]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        with stl.chat_message("assistant"):
            with stl.spinner("Generating response..."):
                response = stl.session_state['rag_chain'].invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })
                stl.markdown(response)

        stl.session_state["messages"].append({"role": "assistant", "content": response})



else:
    stl.info("Please load a YouTube video to start chatting.")
