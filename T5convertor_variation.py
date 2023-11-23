import os
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize T5
model_name = "t5-small"
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
#t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)

def t5_summarize(text, model=t5_model, tokenizer=t5_tokenizer):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load the exclusion list from a file if it exists
exclusion_list_file = "exclusion_list.pkl"
if os.path.exists(exclusion_list_file):
    with open(exclusion_list_file, "rb") as f:
        dynamic_exclusion_list = pickle.load(f)
else:
    dynamic_exclusion_list = set()

# Read conversation from text file
file_path = "conversation.txt"
with open(file_path, "r", encoding="utf-8") as file:
    conversation = file.read()

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(conversation)
filtered_words = [w for w in word_tokens if w.lower() not in stop_words and len(w) > 2]

# Update the dynamic exclusion list based on certain criteria
word_freq = nltk.FreqDist(filtered_words)
high_freq_words = {word for word, freq in word_freq.items() if freq > 10}
dynamic_exclusion_list.update(high_freq_words)

# Save the updated exclusion list to a file for future use
with open(exclusion_list_file, "wb") as f:
    pickle.dump(dynamic_exclusion_list, f)

# Split the conversation into sentences
sentences = sent_tokenize(conversation)
tokenized_sentences = [word_tokenize(sent) for sent in sentences]

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(tokenized_sentences)
corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]

# Train the LDA model
lda = LdaModel(corpus, num_topics=6, id2word=dictionary, passes=60)
topics = lda.print_topics(num_words=15)

# Extract the topic
topic_words = [word for word in topics[0][1].split('"')[1::2]]
excluded_items = stop_words.union(set(string.punctuation)).union(dynamic_exclusion_list)
topic = ' '.join(word for word in topic_words if word.lower() not in excluded_items)

# Summarize the conversation using T5
chunks = [conversation[i:i+512] for i in range(0, len(conversation), 512)]
summaries = [t5_summarize(chunk) for chunk in chunks]
final_summary = ' '.join(summaries)

# Identify action items and dates
action_items = {"Alex": [], "Jamie": []}
keywords = ["you'll","you'll handle", "I'll","I'll handle", "handle", "take on", "work on", "coordinate", "set up", "present", "launch", "monitor", "use", "meet", "track", "address", "integrate", "share", "add", "give", "i need to"]

for sentence in sentences:
    if any(keyword in sentence for keyword in keywords):
        parts = sentence.split(".")
        action = parts[0].strip()
        
        if "Alex" in sentence:
            responsible = "Alex"
        else:
            responsible = "Jamie"
        
        action_items[responsible].append(action)

print(f"Topic: {topic}")
print(f"\nSummary: {final_summary}")
#print(f"Action Items: {action_items}")

print("Alex's Tasks:")
for task in action_items["Alex"]:
    print(f"- {task}")

print("\nJamie's Tasks:")
for task in action_items["Jamie"]:
    print(f"- {task}")
