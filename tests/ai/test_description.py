import os
import openai
from ai_web_explorer import promptrepo


def ask_question(client: openai.OpenAI, text: str, question: str) -> bool:
    system_message = """
    You are a question answering agent. The user asks you a yes or no question 
    question about a given text. You answer with a simple Yes or No. 
    """

    user_message = f"""
    Here is a piece of text:
    ----- TEXT START -----
    {text}
    ----- TEXT END -----

    Anwer the following question with a Yes or No:
    {question}
    """

    messsages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messsages,  # type: ignore
        temperature=0.0,
    )

    if completion.choices[0].message.content is None:
        return False

    answer = completion.choices[0].message.content.lower()
    return "yes" in answer


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "hackernews"
)


def test_description():
    client = openai.Client()

    with open(os.path.join(DATA_DIR, "4.html")) as fd:
        html = fd.read()

    prompt = promptrepo.get_prompt("describe_html")
    result = prompt.execute_prompt(client, html_part=html)
    
    assert len(result.choices) > 0
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) > 0

    description = result.choices[0].message.tool_calls[0].function.arguments
    assert ask_question(client, description, "Does the text mention a login form?")
    assert ask_question(client, description, "Does the text mention a registration form?")
    assert ask_question(client, description, "Does the text mention a reset password link?")
