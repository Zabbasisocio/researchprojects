from transformers import T5ForConditionalGeneration, T5Tokenizer
import datetime

# Initialize T5 tokenizer and model with explicit model_max_length and legacy behavior set to False
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Specify the file path and read the conversation from the text file
file_path = "conversation.txt"
with open(file_path, "r", encoding="utf-8") as file:
    conversation = file.read()

# Split the conversation into chunks of approximately 400 words each
chunks = [conversation[i:i+400] for i in range(0, len(conversation), 400)]

# Summarize each chunk using T5 and concatenate the results
summaries = []
for chunk in chunks:
    input_text = "summarize: " + chunk
    input_tokenized = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(input_tokenized, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(chunk_summary)

final_summary = ' '.join(summaries)

# Generate the output file name
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file_name = f"{current_time}_T5_{model_name}.txt"

# Save the summary to the output file
with open(output_file_name, "w", encoding="utf-8") as output_file:
    output_file.write(final_summary)

print(f"Summary saved to: {output_file_name}")
