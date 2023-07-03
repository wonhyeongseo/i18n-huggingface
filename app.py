import subprocess
import requests
import string
import time
import re

import openai
import gradio as gr

def get_content(filepath: str) -> str:
    url = string.Template(
        "https://raw.githubusercontent.com/huggingface/"
        "transformers/main/docs/source/en/$filepath"
    ).safe_substitute(filepath=filepath)
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        return content
    else:
        raise ValueError("Failed to retrieve content from the URL.", url)

def preprocess_content(content: str) -> str:
    # Extract text to translate from document

    ## ignore top license comment
    to_translate = content[content.find('#'):]
    ## remove code blocks from text
    to_translate = re.sub(r'```.*?```', '', to_translate, flags=re.DOTALL)
    ## remove markdown tables from text
    to_translate = re.sub(r'^\|.*\|$\n?', '', to_translate, flags=re.MULTILINE)
    ## remove empty lines from text
    to_translate = re.sub(r'\n\n+', '\n\n', to_translate)

    return to_translate

def get_full_prompt(language: str, filepath: str) -> str:
    content = get_content(filepath)
    to_translate = preprocess_content(content)

    prompt = string.Template(
        "What do these sentences about Hugging Face Transformers "
        "(a machine learning library) mean in $language? "
        "Please do not translate the word after a 🤗 emoji "
        "as it is a product name.\n```md"
    ).safe_substitute(language=language)
    return '\n'.join([prompt, to_translate.strip(), "```"])

def split_markdown_sections(markdown: str) -> list:
    # Find all titles using regular expressions
    return re.split(r'^(#+\s+)(.*)$', markdown, flags=re.MULTILINE)[1:]
    # format is like [level, title, content, level, title, content, ...]

def get_anchors(divided: list) -> list:
    anchors = []
    # from https://github.com/huggingface/doc-builder/blob/01b262bae90d66e1150cdbf58c83c02733ed4366/src/doc_builder/build_doc.py#L300-L302
    for title in divided[1::3]:
        anchor = re.sub(r"[^a-z0-9\s]+", "", title.lower())
        anchor = re.sub(r"\s{2,}", " ", anchor.strip()).replace(" ", "-")
        anchors.append(f"[[{anchor}]]")
    return anchors

def make_scaffold(content: str, to_translate: str) -> string.Template:
    scaffold = content
    for i, text in enumerate(to_translate.split('\n\n')):
        scaffold = scaffold.replace(text, f'$hf_i18n_placeholder{i}', 1)
    return string.Template(scaffold)

def fill_scaffold(filepath: str, translated: str) -> list[str]:
    content = get_content(filepath)
    to_translate = preprocess_content(content)

    scaffold = make_scaffold(content, to_translate)
    divided = split_markdown_sections(to_translate)
    anchors = get_anchors(divided)

    translated = split_markdown_sections(translated)
    translated[1::3] = [
        f"{korean_title} {anchors[i]}"
        for i, korean_title in enumerate(translated[1::3])
    ]
    translated = ''.join([
        ''.join(translated[i*3:i*3+3])
        for i in range(len(translated) // 3)
    ]).split('\n\n')
    translated_doc = scaffold.safe_substitute({
        f"hf_i18n_placeholder{i}": text
        for i, text in enumerate(translated)
    })

    return [content, translated_doc]

def translate_openai(language: str, filepath: str, api_key: str) -> list[str]:
    content = get_content(filepath)
    return [content, "Please use the web UI for now."]
    raise NotImplementedError("Currently debugging output.")

    openai.api_key = api_key
    prompt = string.Template(
        "What do these sentences about Hugging Face Transformers "
        "(a machine learning library) mean in $language? "
        "Please do not translate the word after a 🤗 emoji "
        "as it is a product name.\n```md"
    ).safe_substitute(language=language)
    
    to_translate = preprocess_content(content)

    scaffold = make_scaffold(content, to_translate)
    divided = split_markdown_sections(to_translate)
    anchors = get_anchors(divided)

    sections = [''.join(divided[i*3:i*3+3]) for i in range(len(divided) // 3)]
    reply = []
    
    for i, section in enumerate(sections):
        chat = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": "\n".join([prompt, section, '```'])
                },]
            )
        print(f"{i} out of {len(sections)} complete. Estimated time remaining ~{len(sections) - i} mins")

        reply.append(chat.choices[0].message.content)

    translated = split_markdown_sections('\n\n'.join(reply))
    print(translated[1::3], anchors)
    translated[1::3] = [
        f"{korean_title} {anchors[i]}"
        for i, korean_title in enumerate(translated[1::3])
    ]
    translated = ''.join([
        ''.join(translated[i*3:i*3+3])
        for i in range(len(translated) // 3)
    ]).split('\n\n')
    translated_doc = scaffold.safe_substitute({
        f"hf_i18n_placeholder{i}": text
        for i, text in enumerate(translated)
    })
    return translated_doc

demo = gr.Blocks()

outputs = gr.outputs.Textbox(label="Translation")
with demo:
    gr.Markdown(
        "# HuggingFace i18n \n"
        "## made easy with this demo."
    )
    with gr.Row():
        language_input = gr.inputs.Textbox(
            default="Korean",
            label=" / ".join([
                "Target language", "langue cible",
                "目标语", "Idioma Objetivo",
                "도착어", "língua alvo"
            ])
        )
        filepath_input = gr.inputs.Textbox(
            default="tasks/masked_language_modeling.md",
            label="File path of transformers document"
        )
    with gr.Tabs():
        with gr.TabItem("Web UI"):
            prompt_button = gr.Button("Show Full Prompt", variant="primary")
            # TODO: add with_prompt_checkbox so people can freely use other services such as DeepL or Papago.
            gr.Markdown("1. Copy with the button right-hand side and paste into [chat.openai.com](https://chat.openai.com).")
            prompt_output = gr.Textbox(label="Full Prompt", lines=3, show_copy_button=True)
            # TODO: add check for segments, indicating whether user should add or remove new lines from their input. (gr.Row)
            gr.Markdown("2. After getting the complete translation, remove randomly inserted newlines on your favorite text editor and paste the result below.")
            ui_translated_input = gr.inputs.Textbox(label="Cleaned ChatGPT initial translation")
            fill_button = gr.Button("Fill in scaffold", variant="primary")
        with gr.TabItem("API (Not Implemented)"):
            with gr.Row():
                api_key_input = gr.inputs.Textbox(label="Your OpenAI API Key")
                api_call_button = gr.Button("Translate (Call API)", variant="primary")
    with gr.Row():
        content_output = gr.Textbox(label="Original content", show_copy_button=True)
        final_output = gr.Textbox(label="Draft for review", show_copy_button=True)

    prompt_button.click(get_full_prompt, inputs=[language_input, filepath_input], outputs=prompt_output)
    fill_button.click(fill_scaffold, inputs=[filepath_input, ui_translated_input], outputs=[content_output, final_output])
    api_call_button.click(translate_openai, inputs=[language_input, filepath_input, api_key_input], outputs=[content_output, final_output])

demo.launch()
