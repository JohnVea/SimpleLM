class Tokenizer:
    def __init__(self, vocabulary, vocabSize= 100):
        self.vocabulary = {}
        self.vocabSize = vocabSize
        self.vocabulary["<PAD>"] = 0
        self.vocabulary["<UNK>"] = 1  # Unknown token
        self.vocabulary["<BOS>"] = 2  # Beginning of sentence
        self.vocabulary["<EOS>"] = 3  # End of sentence
        self.vocabulary["<NL>"] = 4  # New line token
        self.vocabulary["<BOW>"] = 5 # Beginning of word
        self.vocabulary["<EOW>"] = 6 # End of word
        self.vocabulary["<User>"] = 7 # User token
        self.vocabulary["<System>"] = 8 # System token
        self.vocabulary["<Assistant>"] = 9 # Assistant token
        self._build_vocabulary(vocabulary)

    def _build_vocabulary(self, vocabulary):
        baseAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_.!~*'()@#&$%^+=,;:<>?/|\\\"[]{}`àáàâäãåæçéèêëíìîïñóòôöõøúùûüýÿÀÁÂÄÃÅÆÇÉÈÊËÍÌÎÏÑÓÒÔÖÕØÚÙÛÜÝßćčďđłńňŕřšťžźżĆČĎĐŁŃŇŘŠŤŽŹŻÀÁÂÄÃÅÆÇÉÈÊËÍÌÎÏÑÓÒÔÖÕØÚÙÛÜÝßćčďđłńňŕřšťžźżĆČĎĐŁŃŇŘŠŤŽŹŻ"
        for i, word in enumerate(baseAlphabet):
            
            if i >= self.vocabSize:
                break
            if(word not in self.vocabulary):
                self.vocabulary[word] = len(self.vocabulary)
                # print(self.vocabulary)
                # print(len(self.vocabulary))
        
        # print(self.vocabulary)
        # print()


        # print("Initial Vocabulary: ", vocabulary)
        
        j = 1
        tempPair = {}
        i = 0
        while len(self.vocabulary) < self.vocabSize:
            iIcrement = 0
            tempPair.clear()
            for i in range((len(vocabulary)+1)-(j+iIcrement)):
                # if(len(vocabulary[i:(j+iIcrement)]) <= 5):
                #     print("Vocab: " + vocabulary[i:(j+iIcrement)])
                # print(j)
                
                #if(vocabulary[i:(j+iIcrement)] != ""):
                if (vocabulary[i:(j+iIcrement)]) in tempPair:
                    tempPair[vocabulary[i:(j+iIcrement)]] += 1
                else:
                    tempPair[vocabulary[i:(j+iIcrement)]] = 1
                i += 1
                iIcrement += 1
            j += 1
            # print("Here    ")
            # print(tempPair)
            #print(max(tempPair, key=tempPair.get))
            k = 0
            # print("lenght of tempPair: ", len(tempPair))
            addedCount = 0
            for k in range(len(tempPair)):
                #print(max(tempPair, key=tempPair.get))
                if(tempPair[max(tempPair, key=tempPair.get)] > 1):
                    # print(max(tempPair, key=tempPair.get))
                    if(len(max(tempPair, key=tempPair.get)) > 1 and max(tempPair, key=tempPair.get) not in self.vocabulary):
                        print(len(self.vocabulary))
                        self.vocabulary[max(tempPair, key=tempPair.get)] = len(self.vocabulary)
                        addedCount += 1
                elif(len(max(tempPair, key=tempPair.get)) == 1 and max(tempPair, key=tempPair.get) not in self.vocabulary):
                    self.vocabulary[max(tempPair, key=tempPair.get)] = len(self.vocabulary)
                    addedCount += 1
                # print("k: ", k)
                del tempPair[max(tempPair, key=tempPair.get)]
                k+= 1
                if tempPair and (addedCount >= 50000 or len(max(tempPair, key=tempPair.get)) >= 8):
                    print(list(tempPair.items())[:5])
                    break
            i = 0
            
            if(j >= len(vocabulary) or len(self.vocabulary) >= self.vocabSize):
                break



        # print()
        # print(self.vocabulary)
        # # print()
        print("Updated Vocabulary: ", vocabulary)
        print("Vocabulary Size: ", len(self.vocabulary))

    def encode(self, text):
        encoded = []
        temp = text.split()
        # print("Text: ", text)
        for word in text.split():
            if word in self.vocabulary:
                encoded.append(self.vocabulary[word])
            else:
                # encoded.append(1)
                startOfWord = 0
                for char in word:
                    if char in self.vocabulary:
                        if startOfWord == 0:
                            encoded.append(self.vocabulary["<BOW>"])
                            startOfWord = 1
                        encoded.append(self.vocabulary[char])
                    else:
                        encoded.append(1)
                if startOfWord == 1:
                    encoded.append(self.vocabulary["<EOW>"])
                    startOfWord = 0
        # print()
        # print("Encoded: ", encoded)
        return encoded
    
    def decode(self, encodedMatrix):
        decoded = []
        # print(self.vocabulary["<UNK>"])

        startOfwWord = 0
        wordToStrip = " "
        for index in encodedMatrix:
            
            if index in self.vocabulary.values():
                if index == self.vocabulary["<BOW>"]:
                    if(startOfwWord == 0):
                        startOfwWord = 1
                        # for indx in index:
                        #     print("Idx: ", indx)
                        #     if indx in self.vocabulary.values():
                        #         if index == self.vocabulary["<EOW>"]:
                        #             startOfWord = 0
                        #         else:
                        #             decoded.append(list(self.vocabulary.keys())[list(self.vocabulary.values()).index(indx)])
                        #     else:
                        #         decoded.append("<UNK>")
                elif index == self.vocabulary["<EOW>"]:
                    startOfwWord = 0
                    if wordToStrip != "":
                        wordToStrip.strip("")
                        decoded.append(wordToStrip)
                        wordToStrip = ""
                
                else:
                    if(startOfwWord == 1):
                        
                        # wordToStrip = "{}{}".format(wordToStrip, list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)])
                        wordToStrip = wordToStrip + list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)]
                    else:
                        decoded.append(list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)])
            else:
                decoded.append("<UNK>")
        return " ".join(decoded)




# tokenizer = Tokenizer(vocabulary="low lowest John the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Wilson James John Robert Michael David William Richard Joseph Mary Patricia Jennifer Linda Elizabeth Barbara Susan Margaret\n Over hill, over dale, Thorough bush, thorough brier, Over park, over pale, Thorough flood, thorough fire! I do wander everywhere, Swifter than the moon's sphere; And I serve the Fairy Queen, To dew her orbs upon the green; The cowslips tall her pensioners be; In their gold coats spots you see; Those be rubies, fairy favours; In those freckles live their savours; I must go seek some dewdrops here, And hang a pearl in every cowslip's ear.", vocabSize=728)

# lowwjohn = tokenizer.encode("low w john is cool And I Built this thing wi bonsoir  merçi<PAD>")

# print("Decoded: " + tokenizer.decode(lowwjohn))

# print()
# print("Vocabulary Size: ", len(tokenizer.vocabulary))
# print("Vocabulary: ", tokenizer.vocabulary)