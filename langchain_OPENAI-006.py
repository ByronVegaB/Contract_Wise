import os
import json
import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from PIL import Image

# Configuración inicial: título y favicon
st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📄",
    layout="wide"
)

# Ruta para almacenar la base de datos Chroma y JSON
CHROMA_DB_DIR = "C:\\Users\\byron\\Documents\\CURSOS\\IASATURDAY\\Contract_Wise\\vector_db"
RESPONSES_FILE = "C:\\Users\\byron\\Documents\\CURSOS\\IASATURDAY\\Contract_Wise\\datos.json"

# Configurar la API Key de OpenAI
def configure_openai_api_key():
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.sidebar.text_input("Introduce tu OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

# Inicializa los embeddings y el almacenamiento vectorial
def initialize_chroma():
    embeddings = OpenAIEmbeddings()
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vectorstore

# Función para agregar datos desde PDFs de forma modular
def add_pdf_to_vectorstore(vectorstore, pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        vectorstore.add_documents(split_docs)
        vectorstore.persist()
        st.success(f"PDF '{pdf_path}' agregado correctamente.")
    except Exception as e:
        st.error(f"Error al añadir el PDF '{pdf_path}': {e}")

# Crear una instancia de consulta
def create_retrieval_qa(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Generar y guardar preguntas y respuestas
def generate_questions_and_save(qa_chain):
    questions = [
        "Codigo del Contrato",
        "Fecha de Inicio",
        "Fecha de Fin",
        "Tipos de Pólizas",
        "Tipos de Multas"
    ]
    responses = {}
    for question in questions:
        response = qa_chain.invoke({"query": question})
        responses[question] = response["result"]
    with open(RESPONSES_FILE, "w") as file:
        json.dump(responses, file)
    return responses

# Función para cargar el logo
def cargar_logo():
    image = Image.open("C:\\Users\\byron\\Documents\\CURSOS\\IASATURDAY\\LANGCHAIN\\Isotipo SQ_1.png")
    image = image.resize((100, 100))
    st.sidebar.image(image, caption="Contrac Wise")
   
    
# Navegación entre páginas
def mostrar_pagina(pagina, subpagina, vectorstore, qa_chain):
    if pagina == "Inicio":
        st.image("C:\\Users\\byron\\Documents\\CURSOS\\IASATURDAY\\LANGCHAIN\\Isotipo Horizontal Blanco.png", caption="", use_container_width=True)
        #st.title("Contract Wise")
        st.write("")
        st.write("""
                 <div style="font-size: 30px; ">
                 Es una herramienta avanzada de gestión de contratos que utiliza inteligencia
        artificial para simplificar y agilizar todo el ciclo de vida contractual. Desde la revisión de cláusulas
        hasta la generación de nuevos términos, Contract Wise se alinea con leyes y regulaciones
        aplicables, garantizando acuerdos justos y equilibrados. Al automatizar procesos complejos y
        reducir errores, esta plataforma permite a los usuarios gestionar contratos con mayor eficiencia y
        precisión, transformando la forma en que las empresas manejan acuerdos legales.
    </div>""", unsafe_allow_html=True)
        
    elif pagina == "APP":
        if subpagina == "Añadir PDFs":
            st.title("Añadir PDFs")
            uploaded_files = st.file_uploader("Sube tus archivos PDF:", type=["pdf"], accept_multiple_files=True)
            if st.button("Agregar PDFs"):
                for uploaded_file in uploaded_files:
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.read())
                    add_pdf_to_vectorstore(vectorstore, uploaded_file.name)
        elif subpagina == "Datos del Contrato":
            st.title("Datos del Contrato")
            if os.path.exists(RESPONSES_FILE):
                with open(RESPONSES_FILE, "r") as file:
                    responses = json.load(file)
            else:
                responses = generate_questions_and_save(qa_chain)
            st.table(responses)
        elif subpagina == "Hacer una consulta":
            st.title("Hacer una consulta")
            query = st.text_input("Escribe tu pregunta:")
            if st.button("Consultar"):
                if query:
                    result = qa_chain.invoke({"query": query})
                    st.write("### Respuesta:")
                    st.write(result["result"])
                else:
                    st.warning("Por favor, escribe una pregunta antes de consultar.")
    elif pagina == "Acerca de Nosotros":
        st.title("Acerca de Nosotros")
        st.write("Plataforma desarrollada para la gestión eficiente de contratos.")
        st.write("- Elabaorado: Equipo de MUISIANA & EFAKTO")
        st.write("- Visión: Aprovechar la IA para optimizar procesos.")

# Configurar la API Key de OpenAI
configure_openai_api_key()

# Inicializar la base de datos Chroma
vectorstore = initialize_chroma()
qa_chain = create_retrieval_qa(vectorstore)

# Lado izquierdo: Sidebar con logo y menú
with st.sidebar:
    cargar_logo()
    seleccion = st.radio("Navegación", ["Inicio", "APP", "Acerca de Nosotros"])
    subseleccion = None
    if seleccion == "APP":
        subseleccion = st.selectbox("Elige una función:", ["Añadir PDFs", "Datos del Contrato", "Hacer una consulta"])

# Mostrar la página seleccionada
mostrar_pagina(seleccion, subseleccion, vectorstore, qa_chain)
