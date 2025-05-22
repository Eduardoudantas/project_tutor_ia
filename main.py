import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
import torch 
print(torch.cuda.is_available())
# define path to index storage 

FAISS_INDEX_PATH = "faiss_index"


# Function to stream all documents in proper format
def stream_documents():
    for file_path in os.listdir("data"):
        full_path = os.path.join("data", file_path)
        
        if file_path.endswith(".pdf"):
            yield from stream_pdf(full_path)
        elif file_path.endswith(".csv"):
            yield from stream_csv(full_path, chunk_size=500)
        else:
            print(f"Skipping unsupported file: {file_path}")


# Stream PDFs page by page
def stream_pdf(file_path):
    with open(file_path, "rb") as f:
        text = extract_text(f)
        for page in text.split("\n\n"):  # Splitting into pages
            yield Document(page_content=page.strip())  # Convert to LangChain Document


# Stream CSVs row by row
def stream_csv(file_path, chunk_size=500):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield Document(page_content=str(row.to_dict()))  # Convert to LangChain Document


# Uses Ollama embeddings and FAISS to create a vector database dynamically
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                           model_kwargs={'device': 'cuda'})
# Load existing FAISS index or create a new one
if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index from disk.")
else:
    db = FAISS.from_documents(list(stream_documents()), embeddings)  # Embed only if needed
    db.save_local(FAISS_INDEX_PATH)  # Save index for future runs
    print("Created and saved FAISS index.")


#db = FAISS.from_documents(list(stream_documents()), embeddings)  # Convert generator to list

# Creates a function that searches the vector database for the most relevant information 

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

llm = OllamaLLM(model="deepseek-r1:8b", temperature=0)

template = """
Você é um professor universitário de engenharia civil , você é um especialista mas diversas áreas da engenharia civil como estruturas,
materiais, concreto, hidráulica e afins. Você é capaz de responder a perguntas sobre as disciplinas de forma técnica e didática, se embasando 
na formulação matemática e teórica das disciplinas.

Você tem acesso a um banco de dados das provas, livros e apostilas das mais diversas disciplinas da engenharia civil.
{best_practice}

## Siga as seguintes instruções 
1- você deve usar a sua experiência e conhecimento para responder as perguntas dos alunos e ajudar eles a aprender mais sobre o tópico da pergunta, 
 e também deve prover exemplos para facilitar o entendimento. 
2- Ao finalizar sua resposta pergunte se o usuário ainda tem dúvidas sobre o assunto.

3- Só proponha a lista de exercícios se o usuário responder que não possui mais dúvidas sobre o assunto.

##

you MUST use the following format when using equations with mathematical symbols and operations :

''' "$$(EQUATION)$$" '''

this is the only accepted format for equations, where EQUATION  is the equation you want to display, composed of mathematical numbers, symbols and operations.




aqui está o histórico da conversa até agora, use-o para entender o contexto da conversa e responder de forma coerente:
{chat_history}

aqui está uma pergunta do usuário, com base no histório e no seu banco de dados, busque solucionar as dúvidas do usuário: 
{message}

pense passo a passo.

"""

prompt = PromptTemplate(template=template, input_variables=["best_practice", "chat_history", "message"])

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    # controi o histórico da conversa 
    history = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            history += f"Usuário: {msg['content']}\n"
        else:
            history += f"Professor: {msg['content']}\n"
            
    best_practice = "\n".join(retrieve_info(message))
    # gera a resposta com o histórico incluido
    response = chain.run(best_practice=best_practice, chat_history=history, message=message)
    return response

# cria a interface do streamlit 
st.set_page_config(page_title="Tutor(IA)",page_icon=":material/school:")
st.title("Professor")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Digite sua pergunta:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Professor está pensando..."):
        resposta = generate_response(user_input)

    st.session_state.chat_history.append({"role": "assistant", "content": resposta})

# Exibe o chat
for mensagem in st.session_state.chat_history:
    with st.chat_message("user" if mensagem["role"] == "user" else "assistant"):
        st.markdown(mensagem["content"])
            

