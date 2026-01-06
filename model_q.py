# Imported libraries
from langchain_groq import ChatGroq
# from secret_key import openapi_key, groq_apikey
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from Inserting_files import load_pdf_to_strings, load_txt_to_strings
from openai_jac_functions import chunk_research_paper, search_research_db, insert_publications, answer_research_question
# model.py
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

# Set up our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    #cache_folder="./hf_models"   # forces caching locally
    )


# class Product(BaseModel):
#     name: str = Field("The name of the product.")
#     price: float = Field("The product's price.")
#     features: List[str] = Field("Product's features.") 
#     category: str = Field("Product category. One of [Beverages, Dairy, Grocery]")

# Initialize the model
llm_gpt = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

llm_groq = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=groq_api_key
        )

# Initialize LLM and get answer
def run_query_groq(query:str):
    """
    Run a students query against the RAG system.

    Args:
        query (str): User's question.
    Returns:
        dict: AI answer and source metadata.
    """
    answer, sources = answer_research_question(
        query,
        collection, 
        embeddings, 
        llm_groq
    )

    print("AI Answer:", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")

# Initialize LLM and get answer
def run_query_gpt(query:str):
    """
    Run a students query against the RAG system.

    Args:
        query (str): User's question.
    Returns:
        dict: AI answer and source metadata.
    """
    answer, sources = answer_research_question(
        query,
        collection, 
        embeddings, 
        llm_gpt
    )

    print("AI Answer:", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")

def main():
    # run_query_gpt("What what did early scientists contribute to genetics ")
    run_query_gpt("How to cook")

if __name__ == "__main__":
    main()
