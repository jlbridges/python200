import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load the OpenAI API key from .env and confirm it worked
if load_dotenv():
    print("Successfully loaded API key")
else:
    print("Failed to load API key from .env file")

# Make sure the document directory exists before using it
docs_dir = Path(r"C:\Users\jacob\Documents\ctd\python\python_homework\python200\repo\python-200-v1\lessons\06_AI_augmentation\resources\groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the documents
docs = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Loaded {len(docs)} documents")
for doc in docs:
    print(doc.metadata["file_name"])

# Step 3: Build the index and query engine
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# Step 4: Query the assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    response = query_engine.query(q)
    top = response.source_nodes[0]
    print(f"\nQ: {q}")
    print("A:", response)
    print(f"Top source: {top.node.metadata['file_name']}")
    print(f"Similarity Score: {top.score:.4f}")
    print(f"Text: {top.node.get_content()[:200]}...")
    print("-" * 30)

# Reflection:
# All answers sounded confident and were correct. The top chunk was the wrong
# file for weekend hours and dairy-free milk, but the right info was in the
# other chunks. The loyalty score (0.30) was low, which surprised me.

# Step 5: Find a failure
q = "Do you offer delivery?"

response = query_engine.query(q)

print(f"\nQ: {q}")
print("A:", response)
for i, node_with_score in enumerate(response.source_nodes, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text: {node_with_score.node.get_content()[:200]}...")

# Asked about delivery because it's not in any document, so retrieval has
# nothing good to find.
# Went wrong: chunks had low scores (0.31-0.38) and none mention delivery.
# Tone: the model said it had no info instead of guessing, so it can say "I
# don't know", but I still wouldn't trust a confident answer without checking.
# Change: add delivery info and tell the model to say "I don't know" if the
# chunks don't have the answer.

# Step 6: Reflection
# 1. LlamaIndex took about 3 lines plus Settings. Keyword RAG needed several
# functions and a loop, so a framework saves a lot of work.
# 2. A university financial aid office: lots of PDFs that change every year.
# 3. The model can still guess when the answer isn't in the chunks, and it
# sounds just as sure.