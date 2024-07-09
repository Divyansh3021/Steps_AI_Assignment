import streamlit as st
from main import *

'''
Uncomment these 5 lines if running for the first time.
'''

# scraper = NvidiaDocsSpider()
# scraper.run()
# scraper.generate_embeddings()
# store = Storage()
# store.store(scraper.chunks)
# retr = Retriever()

st.title("Steps AI Chatbot")

query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        answer = retr.generate_answer(query)
        st.write("### Answer:")
        st.write(answer)
    else:
        st.write("Please enter a query.")
