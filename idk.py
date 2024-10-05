from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdfplumber

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cpu"

model.to(device)


def query_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def extract_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def query_rag(prompt, path):
    article = extract_pdf(path)
    full_prompt = f"{article}\n\nUser query: {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

current_mode = "llm"

def switch_mode():
    global current_mode
    if current_mode =="llm":
        current_mode = "rag"
    else:
        current_mode = "llm"

while True:
    user_input = input("Enter query (type 'switch' to change mode or 'mode' to check mode): ")
    if user_input.lower() == "switch":
        switch_mode()
        print(f"mode switched to: {current_mode}")
    elif user_input.lower() == "mode":
        print(current_mode)
    elif user_input.lower() == "exit":
        exit()
    else:
        if current_mode == "llm":
            print(query_llm(user_input))
        elif current_mode == "rag":
            print(query_rag("Can you summarize this article?", user_input)) # User input should be a path to a pdf
