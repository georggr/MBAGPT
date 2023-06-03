'''
This Python script creates a chat interface using the Streamlit library. It uses the OpenAI API to interact with a chatbot model named GPT-3.5-turbo, along with custom handlers to route questions to relevant data sources for context-specific answers.

Here's a step-by-step explanation:

1. **Imports**: The necessary libraries and modules are imported. This includes standard Python libraries like `os` for operating system operations, the OpenAI API library, Streamlit for creating an interactive UI, and some custom-defined modules for embeddings, vector stores, and utility functions.

2. **Setting up OpenAI API**: The OpenAI API key is set up using Streamlit secrets.

3. **UI header**: A header for the web application is set up using Streamlit.

4. **Initializing embeddings**: The `OpenAIEmbeddings` object is initialized.

5. **Loading Databases**: Two databases, "buffett" and "branson", are loaded into `Chroma` vector stores, each with its own retriever object. These databases contain embeddings of texts associated with Warren Buffett and Richard Branson respectively.

6. **Session state initialization**: The history of the chat is stored in a Streamlit session state, which is initialized as an empty list if it doesn't already exist.

7. **construct_messages()**: This function constructs formatted messages from the chat history. It ensures the total number of tokens in the messages doesn't exceed the model's limit.

8. **Handler functions**: Four handler functions are defined: `hormozi_handler()`, `buffett_handler()`, `branson_handler()`, and `other_handler()`. Each handler deals with a specific category of user query, preparing the query and context and returning them in the appropriate format.

9. **route_by_category()**: This function routes the user's query to the correct handler based on the category.

10. **generate_response()**: This function generates a response to the user's query. It classifies the intent of the query, routes it based on its category, constructs a set of messages, ensures they fit the model's token limit, sends them to the API, retrieves the response, and appends the assistant's response to the chat history.

11. **User Input**: A text input field is created for users to enter their prompts. When a new prompt is entered, the `generate_response()` function is triggered.

12. **Displaying chat history**: The chat history is displayed in the Streamlit app. User messages and assistant responses are formatted using HTML templates and written out to the Streamlit interface.

13. **User Input**: The Streamlit `text_input` function is used to create a user interface where the user can input their queries or prompts. The input field contains a placeholder text to guide the user. The `on_change` parameter is set to trigger the `generate_response()` function each time the user inputs a new query.

14. **Displaying Chat History**: The script uses a for loop to iterate over the session state history and display the chat history. It differentiates between user and assistant messages and formats them differently using pre-defined HTML templates, with `st.write` displaying these messages in the Streamlit application. The `unsafe_allow_html` parameter allows the use of HTML within the message contents.

Below is a breakdown of some of the more complex functions:

**Handler Functions**: These functions (`hormozi_handler`, `buffett_handler`, `branson_handler`, `other_handler`) take a user query as input and prepare it for the chat model. They retrieve relevant documents or context based on the category of the query, format the query and context in the appropriate manner, and return the formatted message.

- `hormozi_handler` performs a semantic search and formats the results.
- `buffett_handler` and `branson_handler` retrieve relevant documents from the Buffett and Branson databases respectively, then prepare the context and format the message.
- `other_handler` simply formats the query in the appropriate message format.

**route_by_category()**: This function takes a user query and a category as input. Based on the category, it routes the query to the appropriate handler function and returns the result.

**generate_response()**: This function manages the overall process of generating a response to the user's query. It appends the user's query to the chat history, classifies the intent of the query, routes the query to the appropriate handler function, constructs the set of messages to be sent to the chat model, ensures the total number of tokens doesn't exceed the model's limit, sends the messages to the OpenAI API, extracts the assistant's response, and appends the response to the chat history.

In sum, this script uses Streamlit to create an interactive chat interface with a GPT-3.5-turbo chat model, providing context-specific responses by routing queries to relevant data sources.



'''


import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from utils import intent_classifier, semantic_search, ensure_fit_tokens, get_page_contents
from prompts import human_template, system_message
from render import user_msg_container_html_template, bot_msg_container_html_template
import openai

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.header("MBAGPT: Chatting with Multiple Data Sources")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load the Buffett and Branson databases
buffettDB = Chroma(persist_directory=os.path.join('db', 'buffett'), embedding_function=embeddings)
buffett_retriever = buffettDB.as_retriever(search_kwargs={"k": 3})

bransonDB = Chroma(persist_directory=os.path.join('db', 'branson'), embedding_function=embeddings)
branson_retriever = bransonDB.as_retriever(search_kwargs={"k": 3})


# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Construct messages from chat history
def construct_messages(history):
    messages = [{"role": "system", "content": system_message}]
    
    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
    return messages


# Define handler functions for each category
def hormozi_handler(query):
    print("Using Hormozi handler...")
    # Perform semantic search and format results
    search_results = semantic_search(query, top_k=3)
    context = ""
    for i, (title, snippet) in enumerate(search_results):
        context += f"Snippet from: {title}\n {snippet}\n\n"

    # Generate human prompt template and convert to API message format
    query_with_context = human_template.format(query=query, context=context)

    # Return formatted message
    return {"role": "user", "content": query_with_context}


def buffett_handler(query):
    print("Using Buffett handler...")
    # Get relevant documents from Buffett's database
    relevant_docs = buffett_retriever.get_relevant_documents(query)

    # Use the provided function to prepare the context
    context = get_page_contents(relevant_docs)

    # Prepare the prompt for GPT-3.5-turbo with the context
    query_with_context = human_template.format(query=query, context=context)

    return {"role": "user", "content": query_with_context}


def branson_handler(query):
    print("Using Branson handler...")
    # Get relevant documents from Branson's database
    relevant_docs = branson_retriever.get_relevant_documents(query)

    # Use the provided function to prepare the context
    context = get_page_contents(relevant_docs)

    # Prepare the prompt for GPT-3.5-turbo with the context
    query_with_context = human_template.format(query=query, context=context)

    return {"role": "user", "content": query_with_context}


def other_handler(query):
    print("Using other handler...")
    # Return the query in the appropriate message format
    return {"role": "user", "content": query}


# Function to route query to correct handler based on category
def route_by_category(query, category):
    if category == "0":
        return hormozi_handler(query)
    elif category == "1":
        return buffett_handler(query)
    elif category == "2":
        return branson_handler(query)
    elif category == "3":
        return other_handler(query)
    else:
        raise ValueError("Invalid category")

# Function to generate response
def generate_response():
    # Append user's query to history
    st.session_state.history.append({
        "message": st.session_state.prompt,
        "is_user": True
    })
    
    # Classify the intent
    category = intent_classifier(st.session_state.prompt)
    
    # Route the query based on category
    new_message = route_by_category(st.session_state.prompt, category)
    
    # Construct messages from chat history
    messages = construct_messages(st.session_state.history)
    
    # Add the new_message to the list of messages before sending it to the API
    messages.append(new_message)
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
    # Call the Chat Completions API with the messages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    # Append assistant's message to history
    st.session_state.history.append({
        "message": assistant_message,
        "is_user": False
    })


# Take user input
st.text_input("Enter your prompt:",
              key="prompt",
              placeholder="e.g. 'How can I diversify my portfolio?'",
              on_change=generate_response
              )

# Display chat history
for message in st.session_state.history:
    if message["is_user"]:
        st.write(user_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
    else:
        st.write(bot_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
