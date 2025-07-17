from tokenizer import Tokenizer
import random
import math
import numpy as np
import pickle

with open("sentences.txt", "r") as file:
    sentences = file.readlines()

file.close()
print("Number of sentences: ", len(sentences))
for i in range(len(sentences)):
    sentences[i] = sentences[i].strip()  # Removes leading and trailing whitespaces

dictio = ""
for line in sentences:
    dictio += line + " "

dictio += "low lowest John the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Wilson James John Robert Michael David William Richard Joseph Mary Patricia Jennifer Linda Elizabeth Barbara Susan Margaret\n Over hill, over dale, Thorough bush, thorough brier, Over park, over pale, Thorough flood, thorough fire! I do wander everywhere, Swifter than the moon's sphere; And I serve the Fairy Queen, To dew her orbs upon the green; The cowslips tall her pensioners be; In their gold coats spots you see; Those be rubies, fairy favours; In those freckles live their savours; I must go seek some dewdrops here, And hang a pearl in every cowslip's ear."
print("Length of dictionary: ", len(dictio))
tokenizer = Tokenizer(vocabulary=dictio, vocabSize=120000)  #15000  1500
vocab_size = len(tokenizer.vocabulary)  
embedding_dim = 50  
print("\n\n\n\n\n\n")
# inputX = input("Input Something")

def create_training_pair(token_ids, max_target_len=50):
    if len(token_ids) < 2:
        return None, None  # not enough tokens to form a pair
    
    X = token_ids[0]  # First token as input
    Y = token_ids[0:1+max_target_len]  # Up to 49 following tokens as target

    # Optionally pad Y to ensure fixed size
    if len(Y) < max_target_len:
        Y += [0] * (max_target_len - len(Y))  # Pad with 0s if too short

    return X, Y

indx = 0
X = []
Y = []
for pieces in sentences:
    pieces = tokenizer.encode(pieces)
    while len(pieces[indx:]) >= 50:
        x, y = create_training_pair(pieces[indx:indx+50])
        X.append(x)
        Y.append(y)
        indx += 1
    index = 0

print("Number of training pairs: ", len(X))
print("First 5 training pairs: ", X[:5])
print("number of Y: ", len(Y[0]))
print("First 5 Y: ", Y[:5])
print()



# input_tokens = tokenizer.encode("There is a tiny little island in the Arctic called 'Hans' island.")
# input_tokens = tokenizer.encode("Every human spent about half an hour as a single cell.")
# input_tokens = tokenizer.encode("The 57 on Heinz ketchup bottles represents the number of varieties of pickles the company once had.")
# print("Input Tokens: ", input_tokens)
# print("Length of Input Tokens: ", len(input_tokens))

# print(create_training_pair(input_tokens))
print("First 5 sentences: ", sentences[3])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def cross_entropy(predicted, target_index):
    return -np.log(predicted[target_index] + 1e-9)



embedding_matrix = np.random.uniform(0, 0.1, (vocab_size, embedding_dim)) # Embedding matrix setup
print("\n\n\n\n\n\n")
print("Vocabulary Size: ", vocab_size)
print("Embedding Matrix Shape: ", embedding_matrix.shape)
print("Embedding Matrix: \n", embedding_matrix)




learning_rate = 0.1
epochs = 1000 
decay_rate = 0.90 

def softmax(x):
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

# --- Cross-entropy loss ---
def cross_entropy(probs, target_index):
    return -np.log(probs[target_index] + 1e-9)  # small epsilon to avoid log(0)



for epoch in range(epochs):
    total_loss = 0
    for context_token, target_tokens in zip(X, Y):
        # Get vector for context token
        context_vector = embedding_matrix[context_token]  # shape: (50,)

        # Predict scores
        logits = np.dot(embedding_matrix, context_vector)  # (vocab_size,)
        probs = softmax(logits)

        # Compute average loss for this group
        # loss = sum(cross_entropy(probs, t) for t in target_tokens)
        loss = sum(cross_entropy(probs, t) for t in target_tokens) / len(target_tokens)

        total_loss += loss

        # Gradient of loss w.r.t logits
        grad_logits = probs.copy()
        for t in target_tokens:
            grad_logits[t] -= 1 / len(target_tokens)  # average gradient

        # Save current state before update
        old_embedding = embedding_matrix.copy()

        # Update output embeddings
        for i in range(vocab_size):
            embedding_matrix[i] -= learning_rate * grad_logits[i] * context_vector

        # Backprop to input embedding
        grad_context = np.dot(grad_logits, old_embedding)  # shape: (50,)
        embedding_matrix[context_token] -= learning_rate * grad_context

    avg_loss = total_loss / len(X)
    learning_rate = learning_rate * (decay_rate ** (epoch // 100))
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

np.save("embeddingMatrixLookup.npy", embedding_matrix)  # Save embedding matrix 

# Saving tokenizer for later use
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)  


