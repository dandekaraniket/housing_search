import streamlit as st
from PIL import Image
import requests
import json
import time
from housing_search import config
return_flag = config['return_flag']

# Backend API URL
API_URL = "http://localhost:8000/search"

st.title("AI Search for Housing Requirements")

# User input field
query = st.chat_input("Enter your search query:")

# When user clicks search button
if query:
    # Print the query in the terminal
    with st.chat_message("user"):
        st.write(query)
    print(f"User query: {query}")  

    # Display the query in the UI
    st.write(f"🔎 Searching for: **{query}**")

    # Send POST request to FastAPI backend
    response = requests.post(API_URL, json={"query": query})

    if response.status_code == 200:
        results = response.json()
        i = 1
        if return_flag == 1:
            st.write(f"**Semantic Search Results:**")
            for result in results['ss_search'][0]:
                st.write(f"**Search Results: {i}**")
                long_text = json.dumps(f"{result[0]}", indent=4)
                st.write(f"**Source Document: {result[1][0]}**")
                st.markdown(long_text)  
                i = i + 1

            st.write(f"**BM25 Search Results:**")
            for result in results['bm25_results'][0]:
                st.write(f"**Search Results: {i}**")
                long_text = json.dumps(f"{result[0]}", indent=4)
                st.write(f"**Source Document: {result[1][0]}**")
                st.markdown(long_text)  
                i = i + 1            

        else:
            st.write(f"**Final Hybrid Search Results:**")
            for result in results['Final_search_results'][0]:
                st.write(f"**Search Results: {i}**")
                long_text = json.dumps(f"{result[0]}", indent=4)
                st.write(f"**Source Document: {result[1][0]}**")
                #st.markdown(long_text)  
                st.markdown(f"""
                   <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
                   <pre style='white-space: pre-wrap; word-wrap: break-word;'>{long_text}</pre>
                   </div>""", unsafe_allow_html=True)
                st.divider()
                i = i + 1

        st.success("Done!")

    else:
        st.error("Error fetching results. Please try again!")
