import json
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
import logging
import tqdm
import re
import os

logging.basicConfig()
logger = logging.getLogger()

def output_cqs(text, prompt, model, tokenizer, new_params, remove_instruction=False):

    instruction = prompt.format(**{'intervention':text})
    inputs = tokenizer(instruction, return_tensors="pt")
    device = model.device
    inputs = inputs.to(device)

    generation_kwargs = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

    outputs = model.generate(**inputs, **generation_kwargs)


    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if remove_instruction:
        try:
            out = out.split('<|assistant|>')[1]
        except IndexError:
            out = out[len(instruction):]

    return out


def structure_output(whole_text):
    print("=== Raw model output ===")
    print(whole_text)
    print("=========================")
    cqs_list = whole_text.split('\n')
    final = []
    valid = []
    not_valid = []
    for cq in cqs_list:
        if re.match(r'.*\?(")? ?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,"\']*)?"?$', cq):
            valid.append(cq)
        else:
            not_valid.append(cq)

    still_not_valid = []
    for text in not_valid:
        new_cqs = re.split(r'\?"', text + 'end')
        if len(new_cqs) > 1:
            for cq in new_cqs[:-1]:
                valid.append(cq+'?\"')
        else:
            still_not_valid.append(text)

    for i, cq in enumerate(valid):
        occurrence = re.search(r'[A-Z]', cq)
        if occurrence:
            final.append(cq[occurrence.start():])
        else:
            continue

    output = []
    if len(final) >= 3:
        for i in [0, 1, 2]:
            output.append({'id':i, 'cq':final[i]})
        return output
    else:
        logger.warning('Missing CQs')

        return 'Missing CQs'


def main():
    # Load prompt and model name from prompt text file
    with open('prompt_few.txt', 'r', encoding='utf-8') as file: # Changing the prompt file name to either zero-shot or few-shot
        first_line = file.readline()
        model_name = first_line.split('=')[1].strip() # Read the first line and extract the model name
        prompt = file.read().strip() # Read the rest of the file as the prompt template

    # using validation.json for data in the data folder
    with open('data/validation.json', 'r', encoding='utf-8') as f:
        data=json.load(f)

    out = {} # Create an empty dictionary to store the output
    new_params = False
    logger.info(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    logger.info(generation_config)

    if 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    remove_instruction = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info('Loaded '+model_name)
    for _,line in tqdm.tqdm(data.items()):
        text = line['intervention']
        cqs_raw = output_cqs(text, prompt, model, tokenizer, new_params, remove_instruction)
        logger.info("Raw model output:\n%s", cqs_raw)

        parsed_questions = structure_output(cqs_raw)
        line["cqs"] = parsed_questions if parsed_questions else "Missing CQs"


        out[line['intervention_id']]=line
            
    # Save the output to a JSON file
    # --- Changing this to work for windows --- #
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_filename = f'output_fewShot_{safe_model_name}.json'

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4)
    logger.info('Saving file to: %s', os.path.abspath(output_filename), flush=True)



if __name__ == "__main__":
    main()