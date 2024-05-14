import streamlit as st
import time
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import os

load_dotenv()

website_url = "https://www.wshd.nl/"

st.set_page_config(page_title=f'Chat with {website_url}')
st.title('Chat with Waterschap Hollandse Delta 🌊 🇳🇱')
st.write(f'Ask me anything about {website_url}')

# Initialize session state variables if they don't already exist
if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = ""
if 'num_sources' not in st.session_state:
    st.session_state['num_sources'] = 2

# Create sidebar with variables initialized to session state
system_prompt = st.sidebar.text_area("System Prompt", value=st.session_state.system_prompt)
num_sources = st.sidebar.number_input("Number of Sources", value=st.session_state.num_sources, min_value=1, max_value=5)

# Save the inputs in session state when the user interacts with the widgets
if system_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = system_prompt

if num_sources != st.session_state.num_sources:
    st.session_state.num_sources = num_sources

# Initialize chat history, memory, and memory_and_summary
if "messages" not in st.session_state:
    st.session_state.messages = []

#-----------------------CHAT-----------------------
def get_retriever(k):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory='vector_db', embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={"k": k})

    return retriever

# Initialize the model parameters
model = "gpt-3.5-turbo"
max_tokens = 500
temperature = 0.2

# Initialize the ChatOpenAI class with the desired parameters
llm = ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)

# Initialize the ConversationSummaryBufferMemory class with the desired parameters
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, return_messages=True)

# Initialize the retriever
retriever = get_retriever(st.session_state.num_sources)

# Initialize the ConversationChain class with the desired parameters
conversation = ConversationChain(
   llm=llm,
   verbose = True,
   memory=memory
)

#--------------------------------------------------

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve the documents
    docs = retriever.get_relevant_documents(prompt)
        
    # Extract the sources and content from the documents
    sources = []
    content = []
    for doc in docs:
        sources.append(doc.metadata['source'])
        content.append(doc.page_content)
        
    # Augment the prompt
    prompt = st.session_state.system_prompt + ":\n " + prompt

    # Get response from chatbot and update memory variables
    response = conversation.predict(input=prompt)

    # Append the sources to the response with proper formatting
    sources_list = "\n\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources)])
    response_with_sources = f"{response}\n\nThese are the sources I found for the problem:\n\n{sources_list}"

    # Simulate chatbot is typing each token
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        displayed_text = ""
        for token in response_with_sources.split():  # Splitting response into words (tokens)
            displayed_text += token + " "  # Add token and a space
            typing_placeholder.markdown(displayed_text)
            time.sleep(0.05)  # Short delay between tokens
        typing_placeholder.markdown(response_with_sources)  # Display full message

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_with_sources})
