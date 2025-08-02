import numpy as np
import pickle
from transformer import Transformer

with open("wikiSentences.txt", "r") as file:
    wikiSentences = file.readlines()
file.close()

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


embeddingMatrix = np.load("embeddingMatrixLookup.npy")  # Load the embedding matrix

Model = Transformer(embeddingMatrix)

learning_rate = 0.1
epochs = 3
for epoch in range(epochs):
    count = 0
    total_loss = 0
    for sentence in wikiSentences:
        # sentence = tokenizer.encode(sentence.strip())
        sentence = sentence.strip()
        if (len(sentence) < 500):
            count += 1
            # print("Sentence: ", sentence)
            print("Sentence: ", sentence)
            combinedSentence = ""
            for i in range(len(sentence.strip().split())-1):
                combinedSentence += sentence.strip().split()[i] + " "
                # print(combinedSentence)
                outputWord = sentence.strip().split()[i+1] + " "
                # print(outputWord)
                if(len(tokenizer.encode(outputWord)) > len(tokenizer.encode(combinedSentence))):
                    for charcter in outputWord:
                        Charc = charcter
                        total_loss += Model.train(tokenizer.encode(combinedSentence), tokenizer.encode(charcter), learning_rate)
                        combinedSentence += charcter
                    # combinedSentence += charcter + " "
                else:
                    total_loss += Model.train(tokenizer.encode(combinedSentence), tokenizer.encode(outputWord), learning_rate)
            combinedSentence += sentence.strip().split()[-1] + " "
            # print(combinedSentence)
        if(len(sentence) < 40 or len(sentence) > 400):
            break
    learning_rate *= 0.1  # Decay learning rate

    # average_loss = total_loss / len(wikiSentences)
    average_loss = total_loss / count
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")




def find_closest_token(vector, embedding_matrix):
    similarities = [np.dot(vector, emb) / (np.linalg.norm(vector) * np.linalg.norm(emb) + 1e-9)
                    for emb in embedding_matrix]
    return np.argmax(similarities)



inputV = "<BOS> Since "
output = Model.forward(tokenizer.encode(inputV))  # Forward pass through the transformer
print("Output Vector Shape: ", output.shape)
print("Output Vector: \n", output)
similarOutput = [find_closest_token(vec, embeddingMatrix) for vec in output]
print("\n\nDecoded Output: ", tokenizer.decode(similarOutput))
while(tokenizer.decode(similarOutput) != "<EOS>"):
    inputV += tokenizer.decode(similarOutput)
    if(len(tokenizer.encode(inputV)) >= 500):
        break
    output = Model.forward(tokenizer.encode(inputV))
    similarOutput = [find_closest_token(vec, embeddingMatrix) for vec in output]
    print("\n\nDecoded Output: ", tokenizer.decode(similarOutput))

output = inputV + tokenizer.decode(similarOutput)
print("\n\nFinal Decoded Output: ", tokenizer.decode(similarOutput))
    