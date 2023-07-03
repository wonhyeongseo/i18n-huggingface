import gradio as gr
import subprocess
import openai
import time
import re

def translate(text_input, openapi_key):
    openai.api_key = openapi_key
    
    # 라이선스 문장 제거
    rm_line = text_input.find('-->')
    text_list = text_input[rm_line+4:].split('\n')
    print(text_list)

    reply = []
    
    for i in range(0,len(text_list),10):
        content = """What do these sentences about Hugging Face Transformers (a machine learning library) mean in Korean? Please do not translate the word after a 🤗 emoji as it is a product name. Please ignore the video and image and translate only the sentences I provided. Ignore the contents of the iframe tag.
                ```md
                %s"""%'\n'.join(text_list[i:i+10])

        chat = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0301", messages=[
            {"role": "system", 
             "content": content},])
            
        print("질문")
        print(content)
        print("응답")
        print(chat.choices[0].message.content)
        
        reply.append(chat.choices[0].message.content)

        time.sleep(30)
        
    return ''.join(reply)

inputs = [
    gr.inputs.Textbox(lines=2, label="Input Open API Key"),
    gr.inputs.File(label="Upload MDX File")
]

outputs = gr.outputs.Textbox(label="Translation")

def translate_with_upload(text, file):
    
    openapi_key = text
    
    if file is not None:
        text_input = ""
        with open(file.name, 'r') as f:
            text_input += f.read()
            text_input += '\n'
        print(text_input)
        # 텍스트에서 코드 블록을 제거합니다.
        text_input = re.sub(r'```.*?```', '', text_input, flags=re.DOTALL)
        
        text_input = re.sub(r'^\|.*\|$\n?', '', text_input, flags=re.MULTILINE)
        
        # 텍스트에서 빈 줄을 제거합니다.
        text_input = re.sub(r'^\n', '', text_input, flags=re.MULTILINE)
        text_input = re.sub(r'\n\n+', '\n\n', text_input)
    else:
        text_input = ""

    return translate(text_input, openapi_key)

prompt_translate = gr.Interface(
    fn=translate_with_upload,
    inputs=inputs,
    outputs=outputs,
    title="ChatGPT Korean Prompt Translation",
    description="Translate your text into Korean using the GPT-3 model.", verbose=True
)

prompt_translate.launch()
