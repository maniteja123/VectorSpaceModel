from collections import Counter as C
from collections import defaultdict as D
from itertools import product as P
from queryStemmer import porter
import math,sys,pickle,os,tkMessageBox
from nltk.stem.porter import PorterStemmer as PS
from Tkinter import *

class Doc:

    def __repr__(self):
        return 'The document '+str(self.doc_id)

    def __init__(self,doc_id,doc_name):
        self.doc_file = doc_id+'.txt'
        self.doc_id = int(doc_id)
        self.doc_name = doc_name

def Jaccard(A , B):
    return float(len(set(A).intersection(set(B))))/len(set(A).union(set(B)))

class Query:
    
    def __repr__(self):
        return 'The query'

    def __init__(self, query):
        self.string = query
        self.vocab = C()
        self.stemmer = PS()
        self.stem_query = self.stemmed(query)
        #l = self.stem_query.split()
        l = query.split()
        for word in l:
            self.vocab.update([word])
        self.query_length = len(l)

    def stemmed(self, query):
        words = query.split()
        stemmed_query = ''
        for word in words:
            stemmed_query += ' '+str(self.stemmer.stem(word))
        return stemmed_query

    def term_frequency(self, word):
        return self.vocab[word]

    def log_term_frequency(self, word):
        tf = self.term_frequency(word)
        return 1 + math.log10(tf) if tf>0 else 0

    def tf_idf_score(self, word):
        return self.log_term_frequency(word)
    
    def normalization(self):
        return math.sqrt(sum(map(lambda x: x**2, [self.log_term_frequency(word) for word in self.vocab])))

class DocStemmer:

    def __repr__(self):
        return 'The normalization of docs'

    def __init__(self, query):
        self.docs = docs
        self.docs_length = len(docs)
        self.normalize()
        self.stemmer = PS()

    def  normalize(self):
        for doc in self.docs:
            with open(sys.path[0]+'\\reuters\\training\\'+doc.doc_file,'w') as f:
                lines = f.read()
                words = lines.split()
                new_words =[]
                for word in words:
                    new_words.append(str(self.stemmer.stem(word)))
                f.write(' '.join(new_words))
            
class Dictionary:

    def __repr__(self):
        return 'The dictionary'

    def __init__(self, docs):
        self.dictionary  = {}
        self.vocabulary = Vocab(docs)
        self.k_index = K_gram_index(self.vocabulary , 2)
        self.docs = docs
        self.build_dictionary()

    def build_dictionary(self):
        for word in self.vocabulary.vocab:
            self.dictionary[word] = set()
        for doc in self.docs:
            with open(sys.path[0]+'\\reuters\\training\\'+doc.doc_file,'r') as f:
                lines = f.read()
                words = lines.split()
                for word in words:
                    self.dictionary[word].add(doc.doc_id)

    def intersection(self, word1 ,word2):
        if word1 in self.dictionary and word2 in self.dictionary :
            return self.dictionary[word1].intersection(self.dictionary[word2])
        return set()
    
    def union(self, word1 ,word2):
        if word1 in self.dictionary and word2 in self.dictionary :
            return self.dictionary[word1].union(self.dictionary[word2])
        elif word1 in self.dictionary :
            return self.dictionary[word1]
        elif word2 in self.dictionary :
            return self.dictionary[word2]
        return set()
    
    def difference(self, word1 ,word2):
        if word1 in self.dictionary and word2 in self.dictionary :
            return self.dictionary[word1].difference(self.dictionary[word2])
        elif word1 in self.dictionary :
            return self.dictionary[word1]
        return set()

    def symmetric_difference(self, word1 ,word2):
        if word1 in self.dictionary and word2 in self.dictionary :
            return self.dictionary[word1].symmetric_difference(self.dictionary[word2])
        elif word1 in self.dictionary :
            return self.dictionary[word1]
        elif word2 in self.dictionary :
            return self.dictionary[word2]
        return set()

class K_gram_index:

    def __repr__(self):
        return 'The k-gram word index'

    def __init__(self , vocab, sublen):
        self.k_index = {}
        for word in vocab.vocab:
            length = len(word)
            for start in range(length-sublen):
                substr = word[start : start+sublen]
                if substr not in self.k_index :
                    self.k_index[substr] = set()
                self.k_index[substr].add(word)

    def intersection(self, substr1 , substr2):
        if substr1 in self.k_index and substr2 in self.k_index:
            return self.k_index[substr1].intersection(self.k_index[substr2])
        return set()

    def union(self, substr1 , substr2):
        if substr1 in self.k_index and substr2 in self.k_index:
            return self.k_index[substr1].union(self.k_index[substr2])
        elif substr1 in self.k_index :
            return self.k_index[substr1]
        elif substr2 in self.k_index :
            return self.k_index[substr2]
        return set()

    def difference(self, substr1 , substr2):
        if substr1 in self.k_index and substr2 in self.k_index:
            return self.k_index[substr1].difference(self.k_index[substr2])
        elif substr1 in self.k_index :
            return self.k_index[substr1]
        return set()

    def symmetric_difference(self, substr1 , substr2):
        if substr1 in self.k_index and substr2 in self.k_index:
            return self.k_index[substr1].symmetric_difference(self.k_index[substr2])
        elif substr1 in self.k_index :
            return self.k_index[substr1]
        elif substr2 in self.k_index :
            return self.k_index[substr2]
        return set()
        
class Term_Document_Matrix:

    def __repr__(self):
        return 'The term document index'

    def __init__(self,docs):
        self.index = {}
        self.vocabulary = Vocab(docs)
        self.docs = docs
        self.docs_length = len(docs)
        self.lengths = {}
        self.build_index()
        self.build_lengths()

    def build_index(self):
        for word in self.vocabulary.vocab :
            self.index[word] = C()
        for doc in self.docs:
            with open(sys.path[0]+'\\reuters\\training-stemmed\\'+doc.doc_file,'r') as f:
                lines = f.read()
                words = lines.split()
                for word in words:
                    self.index[word].update([doc.doc_id])

    def build_lengths(self):
        self.lengths = {doc.doc_id : self.normalization(doc.doc_id) for doc in self.docs}
        print "index done"

    def term_frequency(self, word, document):
        return self.index[word][document]

    def log_term_frequency(self, word, document):
        tf = self.term_frequency(word, document)
        return 1 + math.log10(self.index[word][document]) if tf > 0 else 0

    def inverse_document_frequency(self, word):
        df = len(self.index[word])
        return math.log10(float(self.docs_length)/df)

    def tf_idf_score(self, word, document):
        return self.log_term_frequency(word, document) * self.inverse_document_frequency(word)

    def normalization(self, document):
        return math.sqrt(sum(map(lambda x: x**2, [self.tf_idf_score(word, document) for word in self.vocabulary.vocab])))

class Vocab:

    def __repr__(self):
        return 'The vocabulary'

    def __init__(self,docs):
        self.vocab = C()
        self.docs = docs
        self.build_vocab()
        self.vocab_size = len(self.vocab.keys())
        self.docs_length = len(docs)

    def build_vocab(self):
        for doc in self.docs:
            with open(sys.path[0]+'\\reuters\\training-stemmed\\'+doc.doc_file,'r') as f:
                lines = f.read()
                words = lines.split()
                for word in words:
                    self.vocab.update([word])
        print "vocab done"

class VectorSpaceModel:

    def __repr__(self):
        return "The vector space retrieval model"

    def __init__(self, index):
        self.index = index
        self.docs = self.index.docs
        self.scores = {}

    def cos_similarity(self, query):
        number_of_terms = len(query.vocab)
        scores = {doc.doc_id : 0 for doc in self.docs}
        terms = query.vocab.keys()
        for word in terms:
            weight_term_query = query.tf_idf_score(word)
            if word not in self.index.vocabulary.vocab:
                continue
            termDocs  = self.index.index[word]
            for doc_id in termDocs:
                weight_term_doc = self.index.tf_idf_score(word, doc_id)
                scores[doc_id] += weight_term_doc * weight_term_query
        docScores = sorted(scores.iteritems() , key=lambda x: x[1], reverse = True)
#        print docScores[:5]
        for doc in self.docs:
            scores[doc.doc_id]  /= self.index.lengths[doc.doc_id]
        docScores = sorted(scores.iteritems() , key=lambda x: x[1], reverse = True)
        print docScores[:5]
        return docScores

class ProbabilisticRetrievalModel:

    def __repr__(self):
        return "The probablisic retrieval model - Probabilistic Indexing"

    def __init__(self, vocab, query):
        self.vocabulary = vocab
        self.docs = vocab.docs
        self.query = query
        self.vocab_size = vocab.vocab_size
        self.doc_prob = {doc.doc_id : 0 for doc in vocab.docs}
        self.query_prob = {query : 0 for query in P([0,1],repeat=query.query_length)}

    def estimate_rank(self, query):
        for target in self.targets:
            self.targ_prob[target]=float(self.numbers[target])/self.total
            self.word_prob[target]={}
            for word in self.vocab.keys():
                try:
                    self.word_prob[target][word] = float(self.vocabs[target][word]+1)/(self.count[target]+self.vocabsize)
                except KeyError:
                    self.word_prob[target][word] = float(1)/(self.count[target]+self.vocabsize)
            print len(self.word_prob[target])
        
class Search:

    def __init__(self,vocabulary):
        self.vocab = vocabulary.vocab
        self.vocabsize = vocabulary.vocab_size
        self.word_prob = {}
        self.build_search()

    def build_search(self):
        for target in self.targets:
            self.targ_prob[target]=float(self.numbers[target])/self.total
            self.word_prob[target]={}
            for word in self.vocab.keys():
                try:
                    self.word_prob[target][word] = float(self.vocabs[target][word]+1)/(self.count[target]+self.vocabsize)
                except KeyError:
                    self.word_prob[target][word] = float(1)/(self.count[target]+self.vocabsize)
            print len(self.word_prob[target])

    def test_search(self,doc):
        with open(doc,'r') as f:
            w = f.read().split()
        output = None
        final = -1*sys.maxint
        for target in self.targets:
            prob = 0
            for word in w:
                if word in self.vocab.keys():
                    print target , word , self.word_prob[target][word],prob
                    prob += math.log(self.word_prob[target][word])
            prob += math.log(self.targ_prob[target])
            #print prob
            if prob > final:
                output = target
                final = prob
        print output


class Application(Frame):

    def get_docs(self):
        query_string = self.QUERY.get()
        q = Query(query_string)
        print query_string
        with(open(sys.path[0]+'\\reuters\\training-stemmed\\'+str(v.cos_similarity(q)[0][0])+'.txt','r')) as g:
            print g.read()

    def createWidgets(self):

        self.LABEL= Label(self)
        self.LABEL["text"]= "Query"
        self.LABEL.pack({"side": "left"})

        self.QUERY = Entry(self)
        self.QUERY["bd"] = 5
        self.QUERY.pack({"side": "left"})
        
        self.SEARCH = Button(self)
        self.SEARCH["text"] = "Search",
        self.SEARCH["command"] = self.get_docs
        self.SEARCH.pack({"side": "top"})
        
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack({"side": "bottom"})
        

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.query_string = ""
        self.pack()
        self.createWidgets()

#q = Query("lion is the king")
#print porter(q.string)
#print q.term_frequency('the')
#print q.normalization()
g = open(sys.path[0]+'\\reuters\\documents.txt','r')
documents = []
for doc in g.readlines():
    documents.append(Doc(*doc.split()[:2]))
f  = sys.path[0]+'\\reuters\\Index-new'

if os.path.exists(f):
    Index = pickle.load(open(f,'rb'))
else:
    Index = Term_Document_Matrix(documents)
    pickle.dump(Index, open(f,'wb'))

print "Index built"

#print Index.inverse_document_frequency('the')
#print Index.term_frequency('the',1)
#print Index.log_term_frequency('the',1)
#print Index.normalization(1)
#Diction = Dictionary(documents)
#print Diction.dictionary['the']
#print Diction.k_index.k_index['th']
#print Diction.k_index.symmetric_difference('th','he')
#print Diction.intersection('the','help')
v = VectorSpaceModel(Index)
#print v.cosine_similarity(q)
#v.cos_similarity(q)
#p = ProbabilisticRetrievalModel(Index,Vocab(docs))

##while True:
##    q = Query(str(raw_input()))
##    with(open(sys.path[0]+'\\reuters\\training-stemmed\\'+str(v.cos_similarity(q)[0][0])+'.txt','r')) as g:
##         print g.read()
##
##while True:
##    q = Query(str(raw_input("String :")))
##    docs = v.cos_similarity(q)
##    print docs[:5]

root = Tk()
app = Application(master=root)
app.mainloop()
try:
    tkMessageBox.showinfo("BYE", "Thanks for using")
    root.destroy()
except Exception:
    pass
