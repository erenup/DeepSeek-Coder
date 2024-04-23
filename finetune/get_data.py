from datasets import load_dataset
import json

from finetune_deepseekcoder import EOT_TOKEN, IGNORE_INDEX

def build_instruction_prompt(instruction: str):
        return '''
    You are an AI programming assistant, utilizing the Open Coder model, developed by erenup, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
    {}
    '''.format(instruction.strip()).lstrip()

code_feedback = load_dataset("m-a-p/Code-Feedback", split='train')
preprocessed_feedback = []
for conversation in code_feedback:
    messages = conversation['messages']
    instruction = ""
    for i, message in enumerate(messages):
        if i == 0:
            assert message['role'] == "user"
            instruction = build_instruction_prompt(message['content'])
        else:
            if message['role'] == "user":
                instruction += f"{message['content']}\n"  # build up instruction content for next assistant output
            elif message['role'] == 'assistant':
                entry = {"instruction": instruction, "output": f"{message['content']}\n{EOT_TOKEN}\n"}
                preprocessed_feedback.append(entry)

                instruction += f"{message['content']}\n{EOT_TOKEN}\n"

jsonl_file_path = "./combined_dataset.jsonl"
with open(jsonl_file_path, 'w') as jsonl_file:
    # Write m-a-p/Code-Feedback entries
    for entry in preprocessed_feedback:
        jsonl_file.write(json.dumps(entry) + '\n')
print("JSON lines file has been created.")

