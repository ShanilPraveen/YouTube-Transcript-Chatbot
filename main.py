import streamlit as stl
import langchain_helper as lch
import textwrap

stl.title("YouTube Assistant")

with stl.sidebar:
    with stl.form(key="my_form"):
        url = stl.sidebar.text_area(
            label="YouTube Video URL",
            max_chars=100
        )
        query = stl.sidebar.text_area(
            label="Ask me about the video",
            max_chars=200,
            key="query_input"
        )

        submit_button = stl.form_submit_button(label="Submit")


if url and query:
    db = lch.create_vector_db(url)
    response = lch.get_response_from_query(db, query)
    stl.subheader("Response")
    stl.write(response)