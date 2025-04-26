
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os

#------------------------------------------------------------------------------
# Login into API
#------------------------------------------------------------------------------

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from the environment
key = os.getenv("OPENAI_API_KEY")

if not key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in your .env file.")

client = OpenAI(api_key=key)
print('API Connected!')

#------------------------------------------------------------------------------

def generate_topics(articles, model="gpt-4.1-nano", temperature=0.1, persona=None, system_prompt=None):
    generations = []

    for article in tqdm(articles, desc="Generating topics"):
        # Construct persona prompt
        persona_intro = ""
        if persona == "bull":
            persona_intro = "You are an overly optimistic investor who sees opportunity in every situation."
        elif persona == "bear":
            persona_intro = "You are a deeply skeptical investor who sees risk and danger in market developments."

        # System prompt
        system_msg = system_prompt or "You are a financial analyst summarizing potential economic or market risks from news articles."

        # User prompt
        user_prompt = f"""{persona_intro}
        
Please analyze the following article and list one potential economic or financial **topic or risk factors** that emerge from it. Only 1-3 keywords.

Article:
\"\"\"{article}\"\"\"

Please format your response as:
{{Topic}}
    
    
"""

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=256,  #set this based on how long you expect the output to be!
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ]
        )

        output = response.choices[0].message.content.strip()
        generations.append(output)

    return generations

#------------------------------------------------------------------------------

file_path = "articles.pq"
df_articles = pd.read_parquet(file_path)

articles = df_articles['headline'].tolist()

# Basic generation
df_articles['generated_topics'] = generate_topics(articles, temperature=0.3)

# With a bear persona
df_articles['generated_topics_bear'] = generate_topics(articles, temperature=0.3, persona='bear')

#------------------------------------------------------------------------------
