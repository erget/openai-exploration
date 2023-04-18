import openai
import os
import sys

openai.api_key = os.environ["OPENAI_API_KEY"]

def get_response(conversation_history):
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=conversation_history + "\n\nAI:",
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["END_OF_AI_RESPONSE"],  # Change the stop sequence
    )

    message = completions.choices[0].text.strip()

    if not message:
        message = "I'm sorry, I couldn't generate a meaningful response. Please try again with a different input or provide more context."

    tokens_used = completions.usage["total_tokens"]

    return message, tokens_used


def main():
    print("Welcome to the AI chat! Type 'quit' or 'exit' to end the conversation.")
    total_tokens = 0
    conversation_history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            print(f"Total tokens used: {total_tokens}")
            print("Goodbye!")
            break

        conversation_history += f"\nYou: {user_input}"
        response, tokens_used = get_response(conversation_history)
        total_tokens += tokens_used
        print(f"AI: {response}")
        print(f"Tokens used this turn: {tokens_used}")

        conversation_history += f"\nAI: {response}END_OF_AI_RESPONSE"

if __name__ == "__main__":
    main()
