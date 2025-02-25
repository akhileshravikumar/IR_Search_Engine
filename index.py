import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import nltk
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Global Variables
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

#Parse XML and Preprocess Data
def parse_cranfield(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    documents = {}
    
    for doc in root.findall("doc"):
        docno = doc.find("docno")
        text = doc.find("text")

        if docno.text is not None and text.text is not None:
            doc_id = int(docno.text.strip())
            text_content = text.text.strip().lower()
            documents[doc_id] = preprocess_text(text_content)
        else:
            print(f"Skipping document due to missing 'docno' or 'text' element: {doc}")
    
    return documents

#Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

#Build Inverted Index
def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(int))
    for doc_id, tokens in documents.items():
        for term in tokens:
            inverted_index[term][doc_id] += 1
    return inverted_index

#Compute TF-IDF for VSM
def compute_tfidf(inverted_index, doc_count):
    tfidf = {}
    for term, doc_dict in inverted_index.items():
        idf = math.log(doc_count / (1 + len(doc_dict)))
        tfidf[term] = {doc_id: tf * idf for doc_id, tf in doc_dict.items()}
    return tfidf

#Implement BM25 Scoring
def bm25_score(query_terms, inverted_index, doc_lengths, avg_doc_len, k1=1.5, b=0.75):
    scores = defaultdict(float)
    for term in query_terms:
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log((len(doc_lengths) - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in inverted_index[term].items():
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_len)))
                scores[doc_id] += score
    return scores

#Implement Language Model(Drichlet Smoothing)
def lm_score(query_terms, inverted_index, doc_lengths, mu=2000):
    scores = defaultdict(float)
    total_terms_in_collection = sum(doc_lengths.values())
 
    collection_freq = {}
    for term in query_terms:
        if term in inverted_index:
            collection_freq[term] = sum(inverted_index[term].values())
        else:
            collection_freq[term] = 0

    for doc_id in doc_lengths:
        doc_len = doc_lengths[doc_id]
        score = 0.0
        for term in query_terms:
            cf = collection_freq.get(term, 0)
            if cf == 0:
                continue  # Skip terms not in the collection
        
            tf = inverted_index[term].get(doc_id, 0) if term in inverted_index else 0
            
            # Calculate probability with Dirichlet smoothing
            p_w_C = cf / total_terms_in_collection
            prob = (tf + mu * p_w_C) / (doc_len + mu)
            score += math.log(prob) if prob > 0 else 0
        
        scores[doc_id] = score
    
    return scores

#Implement Vector Space Model (VSM) Scoring
def vsm_score(query_terms, tfidf, doc_lengths):
    scores = defaultdict(float)
    query_vector = Counter(query_terms)
    query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    
    for doc_id in doc_lengths:
        doc_vector = {term: tfidf.get(term, {}).get(doc_id, 0) for term in query_vector}
        doc_norm = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
        dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)
        cosine_sim = dot_product / (query_norm * doc_norm) if query_norm * doc_norm != 0 else 0
        scores[doc_id] = cosine_sim
    return scores

#Query processing
def process_queries(query_file):
    tree = ET.parse(query_file)
    root = tree.getroot()
    queries = {}
    
    for top in root.findall("top"): 
        q_id_elem = top.find("num")   
        q_text_elem = top.find("title") 
        
        if q_id_elem is not None and q_text_elem is not None:
            q_id = int(q_id_elem.text.strip())
            q_text = q_text_elem.text if q_text_elem.text is not None else ""
            q_text = q_text.strip().lower()
            queries[q_id] = preprocess_text(q_text)
        else:
            print(f"Skipping query due to missing 'num' or 'title' element: {top}")
    
    return queries

#Rank Documents and Output in TREC Format
def rank_documents(queries, inverted_index, doc_lengths, avg_doc_len, tfidf, model):
    results = []
    for q_id, q_terms in queries.items():
        if model == "BM25":
            scores = bm25_score(q_terms, inverted_index, doc_lengths, avg_doc_len)
        elif model == "VSM":
            scores = vsm_score(q_terms, tfidf, doc_lengths)
        elif model == "LM":
            scores = lm_score(q_terms, inverted_index, doc_lengths)
        
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            results.append(f"{q_id} Q0 {doc_id} {rank} {score:.4f} {model}")
    return results

#Save Results and Evaluate
def save_results(results, output_file):
    with open(output_file, "w") as f:
        f.write("\n".join(results))

def evaluate_results(qrel_file, results_file):
    # Path to the jtreceval .jar file
    jtreceval_jar = "jtreceval/target/jtreceval-0.0.5-jar-with-dependencies.jar"
    os.system(f"java -jar {jtreceval_jar} {qrel_file} {results_file}")

# Running the Main Pipeline
documents = parse_cranfield("cranfield-trec-dataset/cran.all.1400.xml")
inverted_index = build_inverted_index(documents)
doc_lengths = {doc_id: len(tokens) for doc_id, tokens in documents.items()}
avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths)
tfidf = compute_tfidf(inverted_index, len(doc_lengths))
queries = process_queries("cranfield-trec-dataset/cran.qry.xml")

#Rank documents using BM25, VSM, and LM
bm25_results = rank_documents(queries, inverted_index, doc_lengths, avg_doc_len, tfidf, "BM25")
vsm_results = rank_documents(queries, inverted_index, doc_lengths, avg_doc_len, tfidf, "VSM")
lm_results = rank_documents(queries, inverted_index, doc_lengths, avg_doc_len, tfidf, "LM")

# Save results
save_results(bm25_results, "bm25_results.trec")
save_results(vsm_results, "vsm_results.trec")
save_results(lm_results, "lm_results.trec")

# Evaluate results using trec_eval
evaluate_results("cranfield-trec-dataset/cranqrel.trec.txt", "bm25_results.trec")
print("-------------------------------------------- bm25")
evaluate_results("cranfield-trec-dataset/cranqrel.trec.txt", "vsm_results.trec")
print("-------------------------------------------- vsm")
evaluate_results("cranfield-trec-dataset/cranqrel.trec.txt", "lm_results.trec")
print("-------------------------------------------- lm")