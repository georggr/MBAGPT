import os
import re
import openai
from retrying import retry
from prompts import classification_prompt
import requests
import tiktoken
import streamlit as st

'''
This script primarily facilitates semantic search functionality and text embeddings. 
It uses OpenAI's GPT-3 API and the Pinecone vector database service. The script also integrates Streamlit, an open-source Python library used for creating data apps, and tiktoken, 
a Python library developed by OpenAI to count tokens in a text string without making an API call.
'''

openai.api_key = st.secrets["OPENAI_API_KEY"]
api_key_pinecone = st.secrets["PINECONE_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
pinecone_endpoint = st.secrets["PINECONE_ENDPOINT"]

intent_classifier_pattern = re.compile(r"\b(Category: \d)")

'''
It  makes an API call to OpenAI to obtain embeddings for a given text string using the 'text-embedding-ada-002' model. The embeddings are then returned in an array format
'''
def get_embeddings_openai(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    response = response['data']

    # extract embeddings from responses0
    return [x["embedding"] for x in response]


'''
It accepts a query, converts it into an embedding vector using the get_embeddings_openai function, and uses this vector to perform a semantic search on Pinecone via its REST API. 
The function retrieves and returns a list of document titles and transcripts that match the search query.
'''

def semantic_search(query, **kwargs):
    # Embed the query into a vector
    xq = get_embeddings_openai(query)

    # Call Pinecone's REST API
    url = pinecone_endpoint
    headers = {
        "Api-Key": api_key_pinecone,
        "Content-Type": "application/json"
    }
    body = {
        "vector": xq[0],
        "topK": str(kwargs["top_k"]) if "top_k" in kwargs else "1",
        "includeMetadata": "false" if "include_metadata" in kwargs and not kwargs["include_metadata"] else True
    }
    try:
        res = requests.post(url, json=body, headers=headers)
        res.raise_for_status()  # Raise an exception if the HTTP request returns an error
        res = res.json()
        titles = [r["metadata"]["title"] for r in res["matches"]]
        transcripts = [r["metadata"]["transcript"] for r in res["matches"]]
        return list(zip(titles, transcripts))
    except Exception as e:
        print(f"Error in semantic search: {e}")
        raise


'''
It uses a retrying decorator to ensure the method will keep trying if an exception is raised. 
It makes an API call to OpenAI's ChatCompletion API, which uses the model 'gpt-3.5-turbo' to generate a response based on the given user prompt. 
It then extracts and returns a category value from the response if it starts with "Category: ", otherwise it returns "No category found".
'''
@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=2000)
def intent_classifier(user_prompt):
    prompt = classification_prompt.replace("$PROMPT", user_prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=20
    )
    intent = response['choices'][0]['message']['content']
    if intent.startswith("Category: "):
        category_value = intent[len("Category: "):].strip()
        return category_value
    else:
        return "No category found"


'''
It ses tiktoken to calculate the number of tokens used by a list of messages. 
Different handling and token calculation strategies are used depending on the model. 
Currently, it only supports the 'gpt-3.5-turbo' model, and for other models, it raises a 'NotImplementedError'.
'''
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

'''
It ensures the total tokens in messages are less than or equal to a given maximum (in this case, 4096). 
If the total tokens exceed this maximum, the function will remove the oldest messages until the total tokens fit within the given limit.
'''
def ensure_fit_tokens(messages):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)
    while total_tokens > 4096:
        removed_message = messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    return messages


'''
It iterates over a list of documents, concatenating their contents together and separating each document's contents with "Document #{i}:\n{doc.page_content}\n\n". 
It returns the combined contents as a string.
'''
def get_page_contents(docs):
    contents = ""
    for i, doc in enumerate(docs, 1):
        contents += f"Document #{i}:\n{doc.page_content}\n\n"
    return contents