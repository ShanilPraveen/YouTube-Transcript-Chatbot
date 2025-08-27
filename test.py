from langchain_helper import create_vector_db, get_response_from_query
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_community.document_loaders import YoutubeLoader

def check_functionality(url):
    db = create_vector_db(url)
    response = get_response_from_query(db, "What is the video about?")
    print(response)

if __name__ == "__main__":
    check_functionality("enter the youtube video URL here")
