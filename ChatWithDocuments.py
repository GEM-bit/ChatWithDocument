import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

#Load  file 
def load_document(file):
    import os
    
    name,extension = os.path.splitext(file)
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file} ...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file} ...')
        loader = Docx2txtLoader(file)
        
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file} ...')
        loader = TextLoader(file)
        
    else:
        print('Document format is not supported')
        return None
        
    data=loader.load()
    return data                           

#Break text into chunks. Ideally the chunk size gives text that makes sense to a human as this goves best search results
def chunk_data(data,chunk_size=256,chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,             
        chunk_overlap=chunk_overlap,            
        length_function=len          
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings=OpenAIEmbeddings()
    vector_store=Chroma.from_documents(chunks,embeddings)
    return vector_store

def ask_and_get_answer(vector_store,q,k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm= ChatOpenAI(model='gpt-3.5-turbo',temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':k})
    chain=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=retriever)  #stuff uses all the text in the prompt    
    answer=chain.run(q)
    return answer

### Embedding cost
def calculate_embedding_cost(text):
    import tiktoken
    enc=tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in text])
    #print(f'Total tokens: {total_tokens}' ) 
    #print(f'Tokan cost in R: {total_tokens / 1000 * 0.0001 * 18,76 }')
    return total_tokens,(total_tokens / 1000 * 0.0001 * 18,76)

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# #######################################################################################################################
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv,find_dotenv
    load_dotenv(find_dotenv(),override=True)

    st.image('LogoBlue.png',width=50)
    st.subheader('LLM Question Answering Application :information_desk_person:')

    with st.sidebar:
        api_key=st.text_input('OpenAI API Key: ',type='password')    #If user doesn't have access to dotenv
        if api_key:                  # If user entered key
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file=st.file_uploader('Upload a file',type=["pdf","txt","docx"])
        chunk_size=st.number_input('Chunk size: ',min_value=100,max_value=2048,value=512,on_change=clear_history)
        k=st.number_input('k: ',min_value=1,max_value=20,value=3,on_change=clear_history)
        add_data=st.button('Add data',on_click=clear_history)
        
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name=os.path.join('./',uploaded_file.name)         #Storing in current directory
                with open(file_name,'wb') as f:
                    f.write(bytes_data)
                data=load_document(file_name)

                chunks=chunk_data(data,chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write('Embeddings Cost: R{embedding_cost:.4f}')    #Display with 4 decimal points
                
                vector_store = create_embeddings(chunks)
                st.session_state.vs=vector_store

                st.success('File uploaded, chunked and embedded')

    q = st.text_input('Ask a question about the content of your file: ')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store,q,k=3)
            st.text_area(label='LLM answer: ',value=answer)

            st.divider

            if 'history' not in st.session_state:
                st.session_state.history =''
            value=(f'Q: {q} /nAnswer: (answer)')
            st.session_state.history = f'{value} /n{"-" * 50} /n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History',value=h, key="history", height=400)