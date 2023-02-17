import openai

def gpt3_chat(text):
    openai.api_key = "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"
    model_engine = "text-davinci-003"
    prompt = text
    print(f"prompt:{prompt}")
    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=5,
    )
    response = completion.choices[0].text
    print(response)

if __name__ == '__main__':
    gpt3_chat("你好，我是大帅哥")
