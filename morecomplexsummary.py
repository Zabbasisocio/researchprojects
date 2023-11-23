import os
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization.summarizer import summarize
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
high_freq_words = {word for word, freq in word_freq.items() if freq > 10}  # threshold set to 10 for demonstration
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
# Exclude stopwords, punctuation characters, and dynamic exclusion list
excluded_items = stop_words.union(set(string.punctuation)).union(dynamic_exclusion_list)
topic = ' '.join(word for word in topic_words if word.lower() not in excluded_items)

# Summarize the conversation
summary = summarize(conversation, word_count=150)

# Identify action items and dates
action_items = {}
keywords = ["handle", "take on", "work on", "coordinate", "set up", "present", "launch", "monitor", "use", "meet", "track", "address", "integrate", "share", "add", "give", "we need to"]

for sentence in sentences:
    if any(keyword in sentence for keyword in keywords):
        # Split the sentence based on punctuation to get the action item
        parts = sentence.split(".")
        action = parts[0].strip()
        
        # Identify the person responsible based on the name mentioned in the sentence
        if "Alex" in sentence:
            responsible = "Alex"
        elif "Jamie" in sentence:
            responsible = "Jamie"
        else:
            responsible = "Both"
        
        # Add to the action items dictionary
        action_items[action] = responsible

print(f"Topic: {topic}")
print(f"Summary: {summary}")
print(f"Action Items: {action_items}")
