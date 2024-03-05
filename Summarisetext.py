import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization.summarizer import summarize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read conversation from text file
file_path = "conversation.txt"
with open(file_path, "r", encoding="utf-8") as file:
    conversation = file.read()

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(conversation)
filtered_words = [w for w in word_tokens if w.lower() not in stop_words]

# Identify topic by frequency distribution
freq_dist = nltk.FreqDist(filtered_words)
common_words = freq_dist.most_common(3)
topic = common_words[0][0]  # Taking the most common word as the topic

# Split the conversation into chunks of approximately 400 words each
chunks = [' '.join(word_tokens[i:i+400]) for i in range(0, len(word_tokens), 400)]

# Summarize each chunk using gensim and concatenate the results
summaries = []
for chunk in chunks:
    chunk_summary = summarize(chunk, word_count=200)
    summaries.append(chunk_summary)

final_summary = ' '.join(summaries)

# Identify action items and dates
action_items = {}
sentences = sent_tokenize(conversation)
for sentence in sentences:
    if "will be responsible for" in sentence:
        parts = sentence.split(" will be responsible for ")
        if len(parts) > 1:
            action_items[parts[0]] = parts[1]
    if "deadline" in sentence:
        parts = sentence.split(" is ")
        if len(parts) > 1:
            action_items["Deadline"] = parts[1]

print(f"Topic: {topic}")
print(f"Summary: {final_summary}")
print(f"Action Items: {action_items}")
