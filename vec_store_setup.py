import os
import wikipedia as wiki
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List

class MyEmbeddings:
    """
    A class used to generate embeddings for documents and queries using a specified model.
    Attributes
    ----------
    model : SentenceTransformer
        An instance of the SentenceTransformer model used for generating embeddings.
    Methods
    -------
    embed_documents(texts: List[str]) -> List[List[float]]:
        Generates embeddings for a list of documents and returns them as a list of lists of floats.
    embed_query(query: str) -> List[float]:
        Generates an embedding for a single query and returns it as a list of floats.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()  

    def embed_query(self, query: str) -> List[float]:
        
        embedding = self.model.encode(query)
        return embedding.tolist()

def download_articles():
    """
    Downloads articles from a list of Wikipedia URLs and saves them as text files.
    The function creates a directory named './data/articles' if it does not exist.
    For each URL in the predefined list, it extracts the page title, fetches the
    Wikipedia page content, and saves it to a text file in the './data/articles' directory.
    If the file already exists, it skips downloading and prints a message.
    Dependencies:
        - os
        - tqdm
        - wikipedia as wiki
    Raises:
        wikipedia.exceptions.PageError: If the Wikipedia page does not exist.
        wikipedia.exceptions.DisambiguationError: If the Wikipedia page title is ambiguous.
    """
    
    urls = [
    "https://en.wikipedia.org/wiki/United_Nations_Convention_on_the_Law_of_the_Sea",
    "https://en.wikipedia.org/wiki/Convention_on_the_High_Seas",
    "https://en.wikipedia.org/wiki/Marine_protected_area",
    "https://en.wikipedia.org/wiki/Environmental_impact_assessment",
    "https://en.wikipedia.org/wiki/Exclusive_economic_zones",
    "https://en.wikipedia.org/wiki/Convention_on_the_Territorial_Sea_and_Contiguous_Zone",
    "https://en.wikipedia.org/wiki/Convention_on_the_Continental_Shelf",
    "https://en.wikipedia.org/wiki/Convention_on_the_High_Seas",
    "https://en.wikipedia.org/wiki/Territorial_sea",
    "https://en.wikipedia.org/wiki/Innocent_passage",
    "https://en.wikipedia.org/wiki/Transit_passage",
    "https://en.wikipedia.org/wiki/Archipelagic_waters",
    "https://en.wikipedia.org/wiki/Baseline_(sea)",
    "https://en.wikipedia.org/wiki/Hot_pursuit",
    "https://en.wikipedia.org/wiki/International_Maritime_Organization",
    "https://en.wikipedia.org/wiki/Port_state",
    "https://en.wikipedia.org/wiki/Dispute_over_the_extended_continental_shelf_in_the_Southern_Zone_Sea_between_Argentina_and_Chile",
    "https://en.wikipedia.org/wiki/Freedom_of_navigation",
    "https://en.wikipedia.org/wiki/Law_of_salvage",
    "https://en.wikipedia.org/wiki/Maritime_Security_Regimes",
    "https://en.wikipedia.org/wiki/Territorial_disputes_in_the_South_China_Sea",  
    "https://en.wikipedia.org/wiki/International_law",
    "https://en.wikipedia.org/wiki/Monism_and_dualism_in_international_law",
    "https://en.wikipedia.org/wiki/Peremptory_norm",
    "https://en.wikipedia.org/wiki/Customary_international_law",
    "https://en.wikipedia.org/wiki/Sources_of_international_law",
    "https://en.wikipedia.org/wiki/Statute_of_the_International_Court_of_Justice",
    "https://en.wikipedia.org/wiki/The_Law_of_Nations",
    "https://en.wikipedia.org/wiki/An_Introduction_to_the_Principles_of_Morals_and_Legislation",
    "https://en.wikipedia.org/wiki/History_of_international_law",
    "https://en.wikipedia.org/wiki/International_humanitarian_law",
    "https://en.wikipedia.org/wiki/Hague_Conventions_of_1899_and_1907",
    "https://en.wikipedia.org/wiki/Nuremberg_trials",
    "https://en.wikipedia.org/wiki/International_Criminal_Tribunal_for_Rwanda",
    "https://en.wikipedia.org/wiki/International_Court_of_Justice",
    "https://en.wikipedia.org/wiki/International_Criminal_Court",
    "https://en.wikipedia.org/wiki/Third_Geneva_Convention",
    "https://en.wikipedia.org/wiki/International_human_rights_law",
    "https://en.wikipedia.org/wiki/Peace_of_Westphalia",
    "https://en.wikipedia.org/wiki/Vienna_Convention_on_the_Law_of_Treaties",
    "https://en.wikipedia.org/wiki/Montevideo_Convention",
    "https://en.wikipedia.org/wiki/United_Nations_General_Assembly",
    "https://en.wikipedia.org/wiki/Universal_Declaration_of_Human_Rights",
    "https://en.wikipedia.org/wiki/Convention_on_the_Privileges_and_Immunities_of_the_United_Nations",
    "https://en.wikipedia.org/wiki/Vienna_Convention_on_the_Law_of_Treaties_Between_States_and_International_Organizations_or_Between_International_Organizations",
    "https://en.wikipedia.org/wiki/Permanent_Court_of_International_Justice",
    "https://en.wikipedia.org/wiki/Hague_Convention_on_Foreign_Judgments_in_Civil_and_Commercial_Matters",
    "https://en.wikipedia.org/wiki/International_Covenant_on_Economic,_Social_and_Cultural_Rights",
    "https://en.wikipedia.org/wiki/International_Convention_on_the_Elimination_of_All_Forms_of_Racial_Discrimination",
    "https://en.wikipedia.org/wiki/Convention_on_the_Elimination_of_All_Forms_of_Discrimination_Against_Women",
    "https://en.wikipedia.org/wiki/United_Nations_Convention_Against_Torture"
]
        
    os.makedirs("./data/articles", exist_ok=True)
    
    for url in tqdm(urls, desc="Downloading articles"):
        page_title = url.split("/")[-1]
        page = wiki.page(page_title)
        
        file_path = f"./data/articles/{page_title}.txt"
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(page.content)
        else:
            print(f"File already exists: {file_path}")
    
def initialize_vectorstore():
    """
    Initializes and returns a Chroma vector store with a specified embedding model.

    This function sets up a Chroma vector store using the "all-MiniLM-L6-v2" embedding model
    from MyEmbeddings. The vector store is configured to persist data in the "./Chromadb" directory.

    Returns:
        Chroma: An instance of the Chroma vector store initialized with the specified embedding model.
    """
    embedding_model = MyEmbeddings("all-MiniLM-L6-v2")
    return Chroma(embedding_function=embedding_model, persist_directory="./Chromadb")

def prepare_documents():
    """
    Prepares and splits text documents into smaller chunks.

    This function performs the following steps:
    1. Initializes a RecursiveCharacterTextSplitter with specified chunk size and overlap.
    2. Loads text files from the specified directory using DirectoryLoader.
    3. Splits the loaded documents into smaller chunks using the text splitter.

    Returns:
        list: A list of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    loader = DirectoryLoader(path="./data/articles/", glob="*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    return chunks

def add_documents_to_vectorstore():
    """
    Adds prepared document chunks to the vector store.

    This function initializes the vector store, prepares the documents by
    splitting them into chunks, and then adds these chunks to the vector store.

    Returns:
        None
    """
    vec_store = initialize_vectorstore()
    download_articles()
    chunks = prepare_documents()
    vec_store.add_documents(chunks)
    
# add_documents_to_vectorstore()