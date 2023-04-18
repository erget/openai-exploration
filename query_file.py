import openai
import sys
import os
import pdfplumber

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages])
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def get_text_from_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python query_file.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Get the text from the file
    content = get_text_from_file(file_path)

    # Check if the content is too large
    max_tokens = 4000
    content_tokens = len(content.split())
    if content_tokens > max_tokens:
        print(f"Error: The content is too large ({content_tokens} tokens). Please provide a smaller file with no more than {max_tokens} tokens.")
        sys.exit(1)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    openai.api_key = openai_api_key

    session_history = content

    print("Starting the chat session. Type 'quit' to end the session.")
    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break

        prompt = f"{session_history}\n\nYou: {query}\nAI:"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

        answer = response.choices[0].text.strip()

        session_history += f"\nYou: {query}\nAI: {answer}"
        print(f"AI: {answer}")
