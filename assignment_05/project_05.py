# project_05.py
# Job Application Helper

# --- Task 1: Setup and System Prompt ---

from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """You are a job application coach for career changers: people moving from one field into a new one who struggle to describe their past experience in language that resonates with employers in their target field.

Your job is to help with job application materials only: rewriting resume bullet points, drafting cover letters, and answering follow-up questions about those materials. If the user asks about something unrelated, politely steer the conversation back to their application.

When rewriting or drafting, emphasize transferable skills and use concrete, action-oriented language. If you need details (target role, past role, accomplishments), ask for them instead of inventing facts. Never fabricate experience, numbers, or credentials.

Always end any rewritten bullets or drafted letters with a reminder that the user should review and edit the output before submitting it anywhere.

You may not know the user's specific industry norms or the expectations of a particular employer, so say so when relevant and remind the user to use their own judgment.

Keep responses concise and professional."""

# Deliberate choice: I told the model never to invent experience, numbers, or
# credentials and to ask for missing details instead. Career changers are
# tempted to stretch their background, and a made-up metric on a resume could
# hurt the user, so the prompt makes the model ask rather than guess.

print("Task 1 - System prompt:")
print(SYSTEM_PROMPT)

# --- Task 2: Bullet Point Rewriter ---

import json


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    Respond ONLY with valid JSON, no other text, no markdown, and no code fences.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages, temperature=0)

    try:
        results = json.loads(response)
    except json.JSONDecodeError:
        print("Task 2 - Response was not valid JSON:", response)
        return []

    for i, item in enumerate(results, start=1):
        print(f"Bullet {i}")
        print("  Original:", item["original"])
        print("  Improved:", item["improved"])

    return results


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time",
]

rewrite_bullets(bullets)

# What makes these bullets weak, and what kinds of changes did the model suggest?
# The bullets are vague: weak verbs ("helped", "made", "worked"), no specifics about
# who, what, or how much, and no results. The model swapped in stronger action verbs,
# added detail about the task, and framed each bullet around an outcome.

# --- Task 3: Cover Letter Generator ---


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Do not invent credentials, employers, or achievements that aren't in the background.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Pandas."

cover_letter = generate_cover_letter(job_title, background)
print("Task 3 - Cover letter opening:")
print(cover_letter)

# Why did you choose those particular examples? What does few-shot help control?
# I kept the two provided examples because they come from different fields (nursing,
# banking) and both connect past experience to the new role in a specific, confident
# way. Few-shot prompting controls tone, length, and structure (past experience, new
# skills, why this company), so the output is specific rather than generic.

# --- Task 4: Moderation Check ---


def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    flagged = result.results[0].flagged

    if flagged:
        print("Sorry, I can't help with that message. Please rephrase and try again.")
        return False
    return True


safe_text = "I want to rewrite my resume bullets for a data analyst role."
flagged_text = "I want to kill my neighbor."

print("Task 4 - Safe input passes:", is_safe(safe_text))
print("Task 4 - Flagged input passes:", is_safe(flagged_text))

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            results = rewrite_bullets(raw_bullets)

            for item in results:
                print("\nOriginal:", item["original"])
                print("Improved:", item["improved"])

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            cover_letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:", cover_letter)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})

            response = get_completion(messages)
            print("\nJob Application Helper:", response)

            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run_chatbot()


# Task 6 - Ethics Reflection
# Option A - Comment block
#
# AI-generated job application advice may favor certain communication styles,
# industries, or cultural backgrounds based on the text it was trained on.
# This could disadvantage applicants whose experience or writing style differs
# from what the model considers professional.
#
# If a job-seeker submits the bot's output without reviewing it, the content
# could misrepresent their experience, include inaccurate information, or fail
# to reflect their actual skills. I would add a disclaimer reminding users to
# review and edit all generated content before submitting it to employers.