import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.openai import OpenAI
from langchain_openai import OpenAI
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
En se basant uniquement sur le {context} qui suit, je souhaite que vous génériez des études de cas stimulantes (des études de cas qui permettent à un étudiant en ingénierie de mettre à l'épreuve ses capacités de réflexion et de résolution de problèmes. Voici la structure que je souhaite que vous suiviez pour chaque étude de cas : Résumé, Problème à résoudre, Solution retenue, Gains attendus, Coûts et moyens engagés, Risques et contraintes) :

{context}

--- 
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)
    # print(prompt)

    # model = Ollama(model="tinyllama")
    model = OpenAI(model="gpt-3.5-turbo-instruct")
    response_text = model.invoke(prompt, max_tokens=-1)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

