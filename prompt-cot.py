import json
from transformers import GenerationConfig, pipeline
import huggingface_hub
import torch
import logging
import tqdm
import re, os, sys, copy, math
from pathlib import Path
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger()
if torch._dynamo.config.cache_size_limit < 20:
    torch._dynamo.config.cache_size_limit = 20
    
def output_cqs(text: str, prompt: list, model: pipeline, new_params: bool, remove_instruction: bool = False, examples: str = None):

    this_prompt = copy.deepcopy(prompt)   
    
    formatting = {'intervention': text}
    formatting.update({'examples': examples}) if examples != None else 0
    for message in this_prompt:
        message["content"][0]["text"] = message["content"][0]["text"].format(**formatting)

    out = model(
        this_prompt,
        max_new_tokens = 512
    )
    out = out[0]["generated_text"]

    if remove_instruction:
        out = out[len(this_prompt):]

    return out


def structure_cot_output(whole_text, return_reasonings = False):
    if type(whole_text) == str:
        lines = whole_text
    else:
        lines = whole_text[0]["content"]

    reasonings = re.findall(r"(?<=REASONING:\s).+\.", lines)
    questions = re.findall(r"(?<=QUESTION:\s).*(?<!REASONING:).*\?", lines)

    output = []
    # only output successfully if there are enough questions (and optionally reasonings) generated
    if len(questions) == 3 and (not return_reasonings or len(reasonings) == 3):
        for i in range(3):
            dict_out = {'id': i, 'cq': questions[i]}
            if return_reasonings:
                dict_out.update({'rs': reasonings[i]})
            output.append(dict_out)
        return output
    else:
        logger.warning('Missing CQs')
        return lines


def structure_examples(ex_dict: dict):
    out = ""
    for _, ex in ex_dict.items():
        ex_str = f"\n### INPUT:\n{ex['intervention']}\n### OUTPUT:\n"
        for answer in ex['cqs']:
            ex_str += f"REASONING: {answer['rs']}\nQUESTION: {answer['cq']}\n"
        out += ex_str
    
    return out

def prompt_data(prompt_type: str, model_line: str, model_size: str, data_slice: str, 
                out_to_file = True, new_params = False, remove_instruction = True, max_tries = 10.0, return_reasonings = False, model = None, annotation: str = ""):

    # using [method]_[model]_[size]_struct.json (Ex. cot_gemma_1b_struct.json) for metadata & message structure of the prompt
    with open(f"prompts/{prompt_type}_{model_line}_{model_size}_struct.json", 'r', encoding = 'utf-8') as struct_file:
        structure = json.load(struct_file)
        try:
            model_name = structure["model"]
        except KeyError:
            logger.warning("No model name found in prompt, defaulting to \"GPT-2\"")
            model_name = "gpt2"
        messages = structure["messages"]
    print(model_name)


    # using [method]_prompt.txt for prompt content (so it can be structured outside limited json requirements)
    with open(f"prompts/{prompt_type}{annotation}_prompt.txt", 'r', encoding = 'utf-8') as prompt_file:
        prompt = ''.join(prompt_file.readlines())
        for message in messages:
            message["content"][0]["text"] = message["content"][0]["text"].format(**{'prompt':prompt})


    # using sample.json for data in the data folder
    with open(f'data/{data_slice}.json', 'r', encoding = 'utf-8') as file:
        data = json.load(file)

    logger.info(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    logger.info(generation_config)

    if model == None:
        # initialise model
        model = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float,
            device_map="auto"
        )


    logger.info('Loaded ' + model_name)
    missing_cq_count = 0
    total_tries = 0

    out = {} # Create an empty dictionary to store the output

    if prompt_type == "acot":
        example_dict = prompt_data("cot", model_line, model_size, "sample", max_tries=math.inf, return_reasonings = True, model = model, out_to_file = False, annotation = annotation)
        examples = structure_examples(example_dict)
    else:   
        examples = None

    # print the prompt without statement for reference in log files
    example_prompt = copy.deepcopy(messages)   
    
    formatting = {'intervention': ''}
    formatting.update({'examples': examples}) if examples != None else 0
    for message in example_prompt:
        message["content"][0]["text"] = message["content"][0]["text"].format(**formatting)

    print(f"PROMPT {prompt_type.upper()}:", example_prompt)

    # generate questions for every statement, retrying a maximum of [max_tries] for each statement
    for _,line in tqdm.tqdm(data.items()):
        text = line['intervention']

        tries = 0
        cqs = ""
        while type(cqs) == str and tries < max_tries:
            cqs = output_cqs(text, messages, model, new_params, remove_instruction, examples)
            cqs = structure_cot_output(cqs, return_reasonings)
            tries += 1
        
        total_tries += tries

        if type(cqs) == str:
            missing_cq_count += 1

        line['cqs'] = cqs
        out[line['intervention_id']] = line

    # Save the output to a JSON file
    now = datetime.now()

    if out_to_file:
        output_name = f"out-{prompt_type}{annotation}_{model_line}-{model_size}_{data_slice}_{now.strftime('%j%H%M')}.json"
        output_path = Path("output", output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(out, f, indent=4)

        logger.info('Saving file to: %s', output_path,flush=True)

        total_cqs = len(data)
        print(f"Successful CQs generated: {total_cqs - missing_cq_count}/{total_cqs} ({((total_cqs - missing_cq_count) / total_cqs)*100:.2f}%)")
        print(f"Total tries: {total_tries}\tTotal retries: {total_tries - total_cqs}")

    return out

def main():

    # get HuggingFace access token from external file
    with open("hf_token.txt") as file:
        access_token = file.read()
    
    huggingface_hub.login(token = access_token)

    # get prompt file attributes
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        logger.error("No filename found")
        return

    model_line = sys.argv[2] if len(sys.argv) > 2 else "gemma"
    model_size = sys.argv[3] if len(sys.argv) > 3 else "1b"
    data_slice = sys.argv[4] if len(sys.argv) > 4 else "sample"
    annotation = sys.argv[5] if len(sys.argv) > 5 else ""

    prompt_data(file_name, model_line, model_size, data_slice, annotation = annotation)

if __name__ == "__main__":
    main()
