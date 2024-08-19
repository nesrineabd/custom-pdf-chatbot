
# Import the needed libraries

from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os


# Get our API key from Openai, an account is needed.

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"


# Read in the pdf file

pdf_reader = PdfReader("input_data.pdf")


# Read data from the file and put them into a string variable called text

text = ''
for i, page in enumerate(pdf_reader.pages):
  tmp_text = page.extract_text()
  if tmp_text:
    text += tmp_text


# Chunk the Documents, the chunks are the building blocs for our LLM
# It go from a simple letter 'g' to a complete word 'data'

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 512,
  chunk_overlap  = 32,
  length_function = len,
  )
texts = text_splitter.split_text(text)


# Initialize the embeddings from the OpenAI library
# It will allow us to give a sementic to our data (contextualization)
# we will use them to convert the chunks of text into vectors

embeddings = OpenAIEmbeddings()


# to finally get a Vector Database (our knowledge base) using the FAISS library and the OpenAI embeddings

docsearch = FAISS.from_texts(texts, embeddings)

# qa_chain is used to connect our similarity search to the prompts–user input
# it can be used for more complex tasks

# The stuff parameter in our qa_chain enables us to build applications like
# this, where documents are small and only a few are passed in for most calls

chain = load_qa_chain(OpenAI(), chain_type="stuff")


# Finally, we get to ask our PDF questions

query = "what is big data ?"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query).strip())

query = "for which engineer profile is this course useful ?"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query).strip())