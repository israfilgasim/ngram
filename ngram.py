import math
from collections import Counter, defaultdict
import decimal

context = decimal.getcontext()
context.prec = 100 


class Ngram:

    def __init__(self, model):

        self.n = model
        self.hashMap = defaultdict(Counter)
        self.uniqueWords = Counter()
        self.allWords = 0



    def trainModel(self, corpus):
        
        for sentence in corpus:
            k = 0 if self.n == 1 else 1
            token = ['<s>'] * (self.n - 1) + sentence + ['</s>'] * k
            self.uniqueWords.update(token)
            self.allWords += len(token)

            for i in range(len(token) - self.n + 1):

                history = tuple(token[i:(i + self.n - 1)])
                nextW = token[i + self.n - 1]

                self.hashMap[history][nextW] += 1

    
    def nextWord(self, words):

        history = tuple(words[-(self.n-1):])

        maxProb = 0
        nextW = ''
        for word, count in self.hashMap[history].items():
           if count > maxProb:
               maxProb = count
               nextW = word
        return nextW
                

    def probability(self, history, word):
        if self.hashMap.get(history) is None:
            return 0
        elif self.hashMap[history].get(word) is None:
            return 0
        else:
            total = sum(self.hashMap[history].values())
            return self.hashMap[history][word] / total if total != 0 else 0

    def perplexity(self, corpus):
        totalprob = decimal.Decimal(1)
        N = 0

        for sentence in corpus:
            token = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(token) - self.n):
                history = tuple(token[i:(i + self.n - 1)])
                word = token[i + self.n - 1]
                prob = self.probability(history, word)
                N += 1
                if prob == 0:
                    return 0
                totalprob *= decimal.Decimal(prob)
            return math.pow((decimal.Decimal(1) / decimal.Decimal(totalprob)), decimal.Decimal(1/N))

    

        
class Smoothing(Ngram):
    def __init__(self, n, smoothingModel = 'None'):
        super().__init__(n)
        self.smoothingModel = smoothingModel

    
    def probability(self, history, word):
        wordCount = self.hashMap[history][word]
        total = sum(self.hashMap[history].values())

        if self.smoothingModel == 'laplace':
            return (wordCount + 1) / (total + len(self.uniqueWords))
        
        elif self.smoothingModel == 'backoff':
            if wordCount != 0:
                return wordCount / total
            elif len(history) > 1:
                return self.probability(history[1:], word)
            else:
                return self.uniqueWords[word] / self.allWords
            
        elif self.smoothingModel == 'interpolation':
            lambdas = [0.1, 0.3, 0.6] 
            interPSum = 0
            for i in range(len(lambdas)):
                subHistory = history[i:]
                subTotal = sum(self.hashMap[subHistory].values())
                subProb = (self.hashMap[subHistory][word] / subTotal) if subTotal != 0 else 0
                interPSum += lambdas[i] * subProb
            return interPSum
        
        elif self.smoothingModel == 'kneser-ney':
            d = 0.75 
            wordCount = max(self.hashMap[history][word] - d, 0)
            total = sum(self.hashMap[history].values())
            contProb = len([1 for hist in self.hashMap if word in self.hashMap[hist]]) / len(self.hashMap)
            if total != 0:
                return (wordCount / total) + (d * len(self.hashMap[history]) / total * contProb)
            else:
                return contProb

        else:
            return super().probability(history, word)

            



    
    