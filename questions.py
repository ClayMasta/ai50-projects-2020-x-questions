import nltk
import os
import sys
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    while True:
        # Prompt user for query
        query = set(tokenize(input("Query: ")))
        
        if query == {'stop'}:
            break

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)
        print()


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for document in os.listdir(directory):
        with open(os.path.join(directory, document), 'r', encoding='utf-8') as f:
            files[document] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    token = nltk.word_tokenize(document)
    filtered = []
    for word in token:
        if word.lower() in nltk.corpus.stopwords.words('english'):
            continue
        elif word[0] in string.punctuation:
            continue
        else:
            filtered.append(word.lower())
    return filtered

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_docs = len(documents)
    idfs = dict()
    for document in documents:
        for word in documents[document]:
            if word not in idfs:
                appearances = 0
                for document in documents:
                    if word in documents[document]:
                        appearances += 1
                idf = math.log(num_docs / appearances)
                idfs[word] = idf
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    filenames = list(files.keys())
    file_tfidfs = dict()
    for file in files:
        total_tfidf = 0
        for word in query:
            if word in idfs:
                tf_idf = files[file].count(word) * idfs[word]
                total_tfidf += tf_idf
        file_tfidfs[file] = total_tfidf
    filenames.sort(key=file_tfidfs.get, reverse=True)
    return filenames[:FILE_MATCHES]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    
    - The returned list of sentences should be of length n and should be ordered with 
    the best match first.
    - Sentences should be ranked according to “matching word measure”: namely, 
    the sum of IDF values for any word in the query that also appears in the sentence. 
    Note that term frequency should not be taken into account here, only inverse 
    document frequency.
    - If two sentences have the same value according to the matching word measure, 
    then sentences with a higher “query term density” should be preferred. 
    Query term density is defined as the proportion of words in the sentence that 
    are also words in the query. For example, if a sentence has 10 words, 3 of 
    which are in the query, then the sentence’s query term density is 0.3.
    - You may assume that n will not be greater than the total number of sentences.
    """
    sentence_idfs = dict()
    sentence_qtds = dict()
    for sentence in sentences:
        total_idf = 0
        found = 0
        for word in query:
            if word in sentences[sentence]:
                total_idf += idfs[word]
                found += 1
        sentence_idfs[sentence] = total_idf
        sentence_qtds[sentence] = found / len(sentence)

    sentence_rank = list(sentence_idfs.keys())
    sentence_rank.sort(key=lambda sentence: (sentence_idfs[sentence], sentence_qtds[sentence]), reverse=True)
    return sentence_rank[:SENTENCE_MATCHES]

if __name__ == "__main__":
    main()
