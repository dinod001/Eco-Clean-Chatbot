# Import necessary libraries
from flask import Flask, render_template, request, jsonify, session

import os
from dotenv import load_dotenv
from pydantic import SecretStr

# Import Pinecone & LangChain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-secret-key-here")  # used to manage user sessions securely

# ---------- Pinecone Setup ----------

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Load Pinecone index (must already be created)
index_name = os.environ.get("PINECONE_INDEX_NAME")

if index_name is None:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set.")
index = pc.Index(index_name)

# ---------- Embeddings + Vector Store Setup ----------

openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

openai_api_key_secret = SecretStr(openai_api_key)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key_secret
)

# Combine Pinecone index with OpenAI embeddings to allow similarity search
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ---------- LLM Setup (GPT-4o) ----------
# Initialize the GPT-4o model with temperature (creativity level)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=1,
    api_key=openai_api_key_secret
)

# ---------- Session Helper Function ----------
def initialize_chat_history():
    """Start session message history if not already started"""
    if 'messages' not in session:
        session['messages'] = []
        session['messages'].append({
            'type': 'system',
            'content': 'You are an assistant for question-answering tasks.'
        })

# ---------- Routes ----------

@app.route('/')
def home():
    """Load main chat interface"""
    initialize_chat_history()
    return render_template('index.html')  # this HTML file should be created in a templates/ folder

@app.route('/chat', methods=['POST'])
def chat():
    """Main endpoint for user messages"""
    initialize_chat_history()
    
    # Get user message from frontend (JSON request)
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Save user message to session
        session['messages'].append({
            'type': 'human',
            'content': user_message
        })
        
        # Create retriever using Pinecone vector search
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",  # only return docs with score > threshold
            search_kwargs={"k": 3, "score_threshold": 0.5}
        )
        
        # Get documents relevant to user's question
        docs = retriever.invoke(user_message)
        docs_text = "".join(d.page_content for d in docs)
        
        # Build a system prompt with the retrieved context
        system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}:"""
        
        system_prompt_fmt = system_prompt.format(context=docs_text)

        # Convert chat history to LangChain format
        langchain_messages = []
        for msg in session['messages']:
            if msg['type'] == 'system':
                langchain_messages.append(SystemMessage(msg['content']))
            elif msg['type'] == 'human':
                langchain_messages.append(HumanMessage(msg['content']))
            elif msg['type'] == 'ai':
                langchain_messages.append(AIMessage(msg['content']))
        
        # Add current context-aware system prompt
        langchain_messages.append(SystemMessage(system_prompt_fmt))
        
        # Call GPT-4o model with all messages
        result = llm.invoke(langchain_messages).content
        
        # Save AI's response to session
        session['messages'].append({
            'type': 'ai',
            'content': result
        })

        # Update session to reflect changes
        session.modified = True
        
        # Send the result back to frontend
        return jsonify({
            'response': result,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({'error': 'An error occurred processing your message'}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history and reset session"""
    session.pop('messages', None)
    initialize_chat_history()
    return jsonify({'success': True})

@app.route('/history')
def get_history():
    """Send chat history to frontend (excluding system message)"""
    initialize_chat_history()
    display_messages = [msg for msg in session['messages'] if msg['type'] != 'system']
    return jsonify({'messages': display_messages})

# ---------- Run Server ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
