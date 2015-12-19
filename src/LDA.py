import numpy as np

alpha = 0.3
eta = 0.3
num_topics = 3 #K

with open("../data/LDA/R3-trn-all.txt") as f:
    documents = f.readlines()
with open("../data/LDA/R3_all_Dictionary.txt") as f:
    vocabulary = f.readlines()


documents = np.asarray(documents)
num_documents = documents.shape[0]
vocabulary = np.asarray(vocabulary)
V = vocabulary.shape[0]
id_by_word = {}

#construct word to id mapping
for id, word in enumerate(vocabulary):
    word = word.rstrip()
    id_by_word[word] = id

#count of words with topic_i in document_n
document_topic_count = np.zeros((num_documents,num_topics))
#sum of all topic-counts, the index is the topic number
document_topic_sum = np.zeros((num_topics,))
topic_term_count = np.zeros((num_topics,V)) #num_topics by num_terms
topic_term_sum = np.zeros((num_topics,)) #a vector of length 3
words_by_topic = np.zeros((V,num_topics))

#Z should be the #docs by #terms in vocabulary
Z = np.zeros((num_documents,))
W = np.zeros((num_documents, V))

#Randomly assigning topics to words
for i, word in enumerate(vocabulary):
    words_by_topic[i] = np.random.multinomial(1,[1/num_topics]*num_topics)

#Build up W matrix
for d, document in enumerate(documents):
    words = document.split()
    for word in words:
        word = word.rstrip()
        word_id = id_by_word[word]
        W[d][word_id] += 1

#Loop through words in the document, increment topics by document
for d, document in enumerate(documents):
    words = document.split()
    word = word.rstrip()
    for word in words:
        word_id = id_by_word[word]
        document_topic_count[d] += words_by_topic[word_id]
        # document_topic_count[d][word_id] += 1

#Sum the total occurrences of a topic
for topic in range(num_topics):
    topic_frequency = np.sum(document_topic_count[:,topic])
    document_topic_sum[topic] = topic_frequency

#Increment topic term count (essentially a frequency of words)
#Go through each document, for each word in the document, look up its ID, look up its topic, then increment its count according to its topic
for document in documents:
    words = document.split()
    for word in words:
        word.rstrip()
        word_id = id_by_word[word]
        topic_of_word = words_by_topic[word_id]
        for t, topic in enumerate(topic_of_word):
            if topic != 0:
                topic_term_count[t][word_id] += 1

#Sum up the terms allocated to each topic
for t in range(num_topics):
    sum = np.sum(topic_term_count[t,:])
    topic_term_sum[t] = sum

not_finished = True
while(not_finished):
    for d, document in enumerate(documents):
        words = document.split()
        for word in words:
            word = word.rstrip()
            #get the word_id
            #Look up the topic assignment of the word
            #decrement from the sums
            word_id = id_by_word[word]
            topic_of_word = words_by_topic[word_id]
            for t, topic in enumerate(topic_of_word):
                if topic != 0:
                    topic_term_sum[t] -= 1
                    document_topic_sum[t] -= 1
                    document_topic_count[d][t] -= 1
                    topic_term_count[t][word_id] -= 1
    not_finished = False

print("Hello, I am a breakpoint. Nice to meet you.")