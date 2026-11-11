from dotenv import load_dotenv
import os
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Concepts Question 1
#
# Scenario A: RAG
# Hundreds of PDFs that change every quarter. With RAG they can just update
# the documents instead of retraining.
#
# Scenario B: Fine-tuning
# They want a specific writing style and have 3,000 examples. Fine-tuning can
# teach the model the voice.
#
# Scenario C: Prompt engineering
# It's one short report, so she can just paste it into the prompt.

# Concepts Question 2
#
# A confident wrong answer is worse because people trust it when it sounds
# sure. If it says "I'm not sure" you'd check it, but if it sounds certain you
# probably won't.
#
# Example: someone asks if two medications are safe to take together and the
# AI confidently says yes when it's not.
#
# The tone matters because the model sounds just as confident when it's wrong,
# so the tone doesn't tell you if it's right.

# Concepts Question 3
#
# steps = [
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Receive the user's query",
#     "Embed the user's query",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]
#
# 1. Extract text from source documents: get the text out of the files.
# 2. Split text into chunks: break the text into smaller pieces.
# 3. Convert text chunks into embeddings: turn each chunk into a vector of numbers.
# 4. Receive the user's query: the user asks their question.
# 5. Embed the user's query: turn the question into a vector too.
# 6. Retrieve the most relevant chunks: find the chunks closest to the question.
# 7. Inject retrieved chunks into the prompt: add the chunks to the prompt with the question.
# 8. Generate a response from the LLM: the model answers using the chunks.


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Question 1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][0])

# It picked loyalty.txt, which is wrong. hours.txt, hiring.txt and loyalty.txt
# all tied at 1 (hours.txt matched "weekends", the other two matched "your").
# The tie got broken by the sort and loyalty.txt came out on top. It matched on
# a random word, not on what the question meant.

# Keyword Question 2
query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][0])

# It returned "None found" because none of the query words are in any document.
# Keyword RAG got it wrong, since menu.txt is the one that should be used but it
# never says "caffeine". Semantic retrieval would do better because it
# compares meaning instead of exact words.

# Keyword Question 3
#
# Prediction: I think it will pick loyalty.txt because rewards sounds like the
# loyalty program.

query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][0])

# I was wrong, it returned "None found". loyalty.txt never uses the word
# "rewards" (or "sign" or "up"), so every score was 0. This surprised me
# because it's obvious to a person, but keyword matching only looks for the
# exact same words.

# Semantic Question 1
#
# 1. A vector embedding is a list of numbers that represents the meaning of a
# piece of text. Similar meanings get similar numbers.
#
# 2. The 0.85 chunk is more relevant. Cosine similarity goes up to 1, so 0.85
# means it's close in meaning to the query and 0.30 means they're only loosely
# related.
#
# 3. Embeddings capture meaning and not exact words, so a chunk can be close to
# the query even if it uses different words (like "fuel efficiency" and "miles
# per gallon").

# Keyword RAG vs. Semantic RAG
#
# | Feature                    | Keyword RAG                       | Semantic RAG                                      |
# |----------------------------|-----------------------------------|---------------------------------------------------|
# | What is compared?          | Exact word overlap                | Embeddings (vectors) of the query and the chunks  |
# | What is retrieved?         | Full document                     | The top-k most relevant chunks                    |
# | Can it handle synonyms?    | No                                | Yes                                               |
# | Storage format             | Plain text dictionary             | Vector store / index of embeddings                |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity (-1 to 1)                       |

# LlamaIndex Question 1
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

docs = SimpleDirectoryReader(r"C:\Users\jacob\Documents\ctd\python\python_homework\python200\repo\python-200-v1\lessons\06_AI_augmentation\resources\brightleaf_pdfs").load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    response = query_engine.query(q)
    print(f"\nQ: {q}")
    print("A:", response)
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Benefits: the top chunk (0.75) was the benefits one, so that's relevant. The
# other two (EcoVolt and the mission overview) weren't. The answer sounded
# confident and had specific stuff like the $600 wellness plan and the 401(k)
# match. No hedging.
#
# Security: the top chunk (0.65) was Network and Data Security, which is
# relevant. The other two were benefits and EcoVolt, which aren't. The answer
# was long and confident.
#
# Unexpected: EcoVolt and other unrelated chunks kept showing up in the top 3.

# LlamaIndex Question 2
q = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    qe = index.as_query_engine(similarity_top_k=k)
    response = qe.query(q)
    print(f"\n=== similarity_top_k={k} ===")
    print("A:", response)
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)

# The answer was almost the same for both. top_k=5 added a few small details
# (like mindfulness sessions) but the scores dropped from 0.75 to 0.42, so the
# extra chunks weren't very relevant. More context isn't always better, one
# chunk was already enough and the extra ones just cost more.

# LlamaIndex Question 3
q = "Who is the CEO of BrightLeaf Solar and how long have they led the company?"

qe = index.as_query_engine(similarity_top_k=3)
response = qe.query(q)

print(f"\nQ: {q}")
print("A:", response)
for i, node_with_score in enumerate(response.source_nodes, 1):
    print(f"\n--- Chunk {i} (score: {node_with_score.score:.4f}) ---")
    print(node_with_score.node.get_content())

# Expected: I thought it would say it didn't know, or maybe make something up.
#
# What happened: it said the info isn't given, so it didn't make anything up.
# The chunks were the mission overview, partnerships and earnings report, none
# of which mention a CEO, but the scores were still kind of high (0.56-0.62).
#
# What I'd change: add a document about the leadership. I'd also use the
# evaluators to check the answers, since a high score doesn't mean the chunk
# has the answer.

# LlamaIndex Question 4
query_engine = index.as_query_engine(similarity_top_k=3)

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

queries = [
    "What employee benefits does BrightLeaf offer?",
    "Who is the CEO of BrightLeaf Solar and how long have they led the company?",
]

for q in queries:
    response = query_engine.query(q)
    faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
    relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
    print(f"\nQ: {q}")
    print("A:", response)
    print("Faithfulness:", faithfulness_result.score)
    print("Relevancy:", relevancy_result.score)

# Faithfulness 1.0 means the answer is backed up by the retrieved chunks and
# isn't made up. 0.0 would mean it has stuff the chunks don't support.
#
# Relevancy is whether the answer actually addresses the question.
# Faithfulness checks the answer against the chunks, relevancy checks it
# against the question.
#
# The scores changed. Benefits got 1.0 and 1.0. The CEO question got 1.0
# faithfulness but 0.0 relevancy. The model said it didn't have the info, so
# it didn't make anything up (faithful), but it didn't answer the question
# (not relevant) because retrieval never found anything about a CEO.
#
# LLM-as-a-judge means another LLM (gpt-4o-mini) looks at the question, answer
# and context and says YES or NO. It's used because the answers are text, so
# you can't just use a normal accuracy score.