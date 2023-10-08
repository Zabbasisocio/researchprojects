import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization.summarizer import summarize
from gensim import corpora
from gensim.models.ldamodel import LdaModel

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
filtered_words = [w for w in word_tokens if w.lower() not in stop_words and len(w) > 3]

# Split the conversation into sentences
sentences = sent_tokenize(conversation)
tokenized_sentences = [word_tokenize(sent) for sent in sentences]

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(tokenized_sentences)
corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]

# Train the LDA model
lda = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
topics = lda.print_topics(num_words=5)

# Extract the topic
topic = ' '.join(word for word in topics[0][1].split('"')[1::2])

# Summarize the conversation
summary = summarize(conversation, word_count=100)

# Identify action items and dates
action_items = {}
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
print(f"Summary: {summary}")
print(f"Action Items: {action_items}")
