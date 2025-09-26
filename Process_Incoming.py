import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Enter your query: ")
query_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()
top_results = 5
max_indices = similarities.argsort()[::-1][0:top_results]

new_df = df.iloc[max_indices]
prompt = f'''I am providing you with some chunks of text from a tutorial video. Here are the video 
subtitle chunks containing video title, video number, start time in second, end time in second
and text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json()}
-------------------------------
"{incoming_query}"
User asked this question relaated to video chunks, you have to answer where and how much content
is taught in which video (in which video and at what time stamp) and guide the uer to go to that 
particular video. If user asks unrelated questions, tell him that you can only answer questions
related to the videos provided.
'''

with open('prompt.txt', 'w', encoding='utf-8') as f:
    f.write(prompt)

response = inference(prompt)["response"]

with open('response.txt', 'w', encoding='utf-8') as f:
    f.write(response)