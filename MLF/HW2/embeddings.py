
#------------------------------------------------------------------------------

"""
This script defines a function to convert text into embeddings using OpenAI’s
embedding API: https://platform.openai.com/docs/guides/embeddings

It uses the `text-embedding-3-small` model, which outputs a 1536-dimensional vector.
This model follows the Matryoshka embedding approach 
(see: https://arxiv.org/abs/2205.13147), which enables flexible slicing of the
embedding vector to lower dimensions—while retaining semantic quality across scales.

For more background, see this overview:
https://medium.com/vector-database/matryoshka-embeddings-detail-at-multiple-scales-15cfad7cdd90
"""

#------------------------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os

#------------------------------------------------------------------------------
# Login into API
#------------------------------------------------------------------------------

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
if not key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in your .env file.")

client = OpenAI(api_key=key)
print('API Connected!')

#------------------------------------------------------------------------------
# This function converts text to embeddings and adds the embedding vector as 
# an addtional vector to your df. You can input the preferred dimension. 
#------------------------------------------------------------------------------

def get_batch_embeddings(texts, model="text-embedding-3-small", dimensions=250, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model,
            dimensions=dimensions
        )
        batch_embeddings = [res.embedding for res in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

#------------------------------------------------------------------------------

file_path = "articles.pq"
df_articles = pd.read_parquet(file_path)

text = df_articles['headline'].tolist()

df_articles['ada_embedding'] = get_batch_embeddings(text, dimensions=250)

#------------------------------------------------------------------------------
