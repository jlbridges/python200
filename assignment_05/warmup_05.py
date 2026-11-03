from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

print("API Q1 - Response text:", response.choices[0].message.content)
print("API Q1 - Model:", response.model)
print("API Q1 - Total tokens used:", response.usage.total_tokens)

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"API Q2 - Temperature {temp}:", response.choices[0].message.content)
# What do you notice about how the outputs differ?
# Temperature 0 gives the same, most predictable name each run. At 0.7 the names
# vary but stay sensible. At 1.5 the output gets random and sometimes odd.
#
# Which temperature for consistent, reproducible output?
# Temperature 0, since it always picks the highest-probability token.

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

for i, choice in enumerate(response.choices, start=1):
    print(f"API Q3 - Completion {i}:", choice.message.content)

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)

print("API Q4 - Response:", response.choices[0].message.content)
print("API Q4 - Finish reason:", response.choices[0].finish_reason)

# What happened, and why use max_tokens in a real application?
# The response was cut off mid-sentence after 15 tokens. In a real app, max_tokens
# caps response length, which keeps costs and latency under control.

# --- System Messages and Personas ---

# System Q1
messages_tutor = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]

response_tutor = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_tutor,
)
print("System Q1 - Encouraging tutor:", response_tutor.choices[0].message.content)

messages_pirate = [
    {
        "role": "system",
        "content": "You are a grumpy pirate captain who explains everything using sea and treasure metaphors.",
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]

response_pirate = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_pirate,
)
print("System Q1 - Grumpy pirate:", response_pirate.choices[0].message.content)

# What changed?
# The question was identical, but the system message changed the tone and style
# of the answer: the tutor was gentle and encouraging, the pirate was gruff and
# metaphor-heavy. The system message sets the persona without changing the question.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q2 - Response:", response.choices[0].message.content)

# Why does the model know Jordan's name, even though it's stateless?
# Because the earlier messages are sent again in this same request. The model has no
# memory between calls; it only sees the conversation history we pass in the list.

# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content
for i, review in enumerate(reviews, start=1):
    prompt = f"""
Classify the sentiment of the review below as positive, negative, or mixed.
Respond with only one word.

Review: '''{review}'''
"""
    result = get_completion(prompt, temperature=0)
    print(f"Prompt Q1 - Review {i}:", result)

# Prompt Q2 - One-Shot
for i, review in enumerate(reviews, start=1):
    prompt = f"""
Classify the sentiment of the review below as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify this review:
Review: "{review}"
Sentiment:"""
    result = get_completion(prompt, temperature=0)
    print(f"Prompt Q2 - Review {i}:", result)

# Did adding one example change the format or consistency compared to Q1?
# The output stayed one clean label per review, and the example set the format
# (just the sentiment word). Results were consistent with Q1 on these easy reviews.

# Prompt Q3 - Few-Shot
for i, review in enumerate(reviews, start=1):
    prompt = f"""
Classify the sentiment of the review below as positive, negative, or mixed.

Example 1:
Review: "Absolutely loved it, would buy again."
Sentiment: positive

Example 2:
Review: "Broke after one day and the refund was denied."
Sentiment: negative

Example 3:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify this review:
Review: "{review}"
Sentiment:"""
    result = get_completion(prompt, temperature=0)
    print(f"Prompt Q3 - Review {i}:", result)

# Zero-shot: simple, clear tasks where speed and cost matter.
# One-shot: when you mainly need to show the output format.
# Few-shot: nuanced tasks (like "mixed") where several examples improve consistency.

# Prompt Q4 - Chain of Thought
prompt = """
Show your step-by-step reasoning, then give the final answer on its own line labeled: Final answer: <value>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
result = get_completion(prompt, temperature=0)
print("Prompt Q4 - Full response:", result)

# Why does step-by-step reasoning improve accuracy?
# The model produces intermediate steps first, so each calculation builds on the previous one
# instead of jumping to an answer. This reduces errors and makes mistakes easy to spot.

# Prompt Q5 - Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review below and respond ONLY with valid JSON.
Keys: sentiment, confidence (a float from 0 to 1), reason (one sentence).

Review: '''{review}'''
"""
result = get_completion(prompt, temperature=0)
print("Prompt Q5 - Raw response:", result)

try:
    data = json.loads(result)
    print("Prompt Q5 - Sentiment:", data["sentiment"])
    print("Prompt Q5 - Confidence:", data["confidence"])
    print("Prompt Q5 - Reason:", data["reason"])
except json.JSONDecodeError:
    print("Prompt Q5 - Response was not valid JSON:", result)

# Prompt Q6 - Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
result = get_completion(prompt, temperature=0)
print("Prompt Q6 - Instructions text:", result)

non_instruction_text = "The Amazon rainforest is the largest tropical rainforest in the world. \
It is home to millions of species of insects, plants, and animals."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""
result = get_completion(prompt, temperature=0)
print("Prompt Q6 - Non-instructions text:", result)

# What problem do delimiters help prevent?
# They keep user-provided text separate from instructions, so the model doesn't
# confuse the data with commands (prompt injection) or misread where the text ends.

# --- Local Models with Ollama ---

# Ollama Q1
prompt = "Explain what a large language model is in two sentences."
result = get_completion(prompt, temperature=0)
print("Ollama Q1 - OpenAI response:", result)

# Ollama output (paste your real output between the quotes below):
"""
Ollama Q1 - OpenAI response: A large language model is an artificial intelligence system designed to understand and generate human-like text by analyzing vast amounts of written data. It uses deep learning techniques, particularly neural networks, to predict and produce coherent and contextually relevant language based on the input it receives.
"""

# What differences did you notice between the two responses?
# <Describe what you actually saw, e.g. the local model's length, accuracy, and
# whether it printed extra "thinking" text before the answer.>
#
# Advantage of running locally: no per-token cost, and your data stays on your machine.
# Disadvantage: small local models are usually less accurate and can be slower on a laptop.