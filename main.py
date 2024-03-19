# LangChain Provideds classess to help load data from different types of files
# These are called loaders

    # facts.txt --> TextLoader
    # report.pdf --> PyPDFLoader
    # users.json --> JSONLoader
    # blog.md --> Unstructured MarkdownLoader

# also provide classes to load up any kind of files from different locations
    # COnfusing
        # S3FileLoader
            # users.json
            # blog.md
            # report.pdf
            # facts.txt

# Note : Many loaders require some dependent dependency

# Result of loading a file is a "Document"
# Contains the contents of the file
    # ("page_content")
# Contains info about where this document came from ("metadata")

# STEP-2
    # Option #1
        # Put entire contents of "facts.txt" file into the prompt
            # Downside: 
                # 1. longer prompt = costs more to run
                # 2. longer prompt = takes longer to run
                # 3. longer prompt = LLM has a harder time finding relevant facts

    # Options #2
        # Put only the top 1-3 relevant facts into the prompt
            # Upside:
                # shorter prompt = Less expensive, faster, more focused
            # Downside:
                # need to somehow find the most relevant facts

# Sematic Searching
    # Understand the goal of the user search
    # We are goind to implement this using embedding

        # Embedding is a list of numbers between -1 and 1 that score how much a piece a text is talking about some particular quality

        # These embedding only rate 2 qualities, but real embedding frequently have 700 to 1500 "dimensions" or qualities that they score
            # Squared L2 -> using the distance between two points to figure out how similar they are
            # Cosine Similarity -> using the angle between two vectors to figure out how similar they are
# How to proceed now ?
"""
    1. Split the text into seperate chumks
    2. Calculate embedding for each chunk
        Embedding Model
            SentanceTransformer
                all-mpnet-base-v2
                    768 dimensions

            OpenAI Embedding
                1536 dimensions
                
    3. Store embedding in a database specialised in storing embeddings - Vector Store
            ChromaDB --> SQLite
    4. Create embedding out of the user's question
    5. Do a similarity search with our stored embeddings to find the ones most similar to the user's question
    6. Put the most relevant 1-3 facts into the prompt along with the user's questions
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# results = db.similarity_search_with_score(
#     "What is an interesting fact about English language ?",
#     k=1
# )

# for result in results:
#     print("\n")
#     print(results[1])
#     print(result[0].page_content)

results = db.similarity_search(
    "What is an interesting fact about English language ?",
    k=1
)

for result in results:
    print("\n")
    print(result.page_content)