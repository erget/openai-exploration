# Heavily inspired from
# https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a
import sys
import time

import numpy as np
import openai
import pandas as pd
import pdfplumber
import tiktoken
from openai.embeddings_utils import distances_from_embeddings


def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    return text


def extract_text_from_pdf():
    global f
    text_from_pdf = extract_pdf_text(pdf_path)
    # Save the extracted text to a file
    with open("data/context/text_from_pdf.txt", "w", encoding="UTF-8") as f:
        f.write(text_from_pdf)


def remove_newlines(series):
    series = series.str.replace('\n', ' ')
    series = series.str.replace('\\n', ' ')
    series = series.str.replace('  ', ' ')
    series = series.str.replace('  ', ' ')
    return series


def scrape_text():
    global df
    # Create the dataframe from the text
    df = pd.DataFrame({"text": [text]})
    # Modify this line to work with the new DataFrame
    df['text'] = remove_newlines(df.text)
    df.to_csv('data/context/scraped.csv')
    df.head()


def generate_embeddings():
    global tokenizer, df
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df = pd.read_csv('data/context/scraped.csv', index_col=0)
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    max_tokens = 500

    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens=max_tokens):

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        # Add the last chunk to the list of chunks
        if chunk:
            chunks.append(". ".join(chunk) + ".")

        return chunks

    shortened = []
    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])
    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    def create_embedding(text):
        try:
            return openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
        except openai.error.RateLimitError as e:
            wait_time = int(e.headers.get("Retry-After", 1))
            print(f"Rate limit reached. Waiting for {wait_time} seconds.")
            time.sleep(wait_time)
            return create_embedding(text)

    start_time = time.time()
    num_requests = 0

    df['embeddings'] = df.text.apply(lambda x: create_embedding(x))

    num_requests += 1

    # If the number of requests is about to reach 60, wait for the remaining time in the minute
    if num_requests == 59:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            wait_time = 60 - elapsed_time
            print(f"Approaching rate limit. Waiting for {wait_time} seconds.")
            time.sleep(wait_time)
    df.to_csv('data/context/embeddings.csv')


def create_context(
        question, df, max_len=1800
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        debug=False,
        max_tokens=150,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on "
                   f"the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_pdf>")
        sys.exit(1)

    # Get the PDF path from the command line argument
    pdf_path = sys.argv[1]

    # Can skip following block if we can use cached results
    # Read the extracted text from the correct file path
    # with open("text_from_pdf.txt", "r", encoding="UTF-8") as f:
    #     text = f.read()
    # extract_text_from_pdf()  # If PDF is already known
    # scrape_text()
    # generate_embeddings()

    df = pd.read_csv('data/context/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    print("Starting an interactive Q&A session...\n")

    while True:
        user_question = input("Ask your question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        else:
            answer = answer_question(df, question=user_question, max_tokens=2000)
            if answer:
                print("\nAnswer:", answer)
            else:
                print("\nI'm sorry, I couldn't find an answer to your question.")
