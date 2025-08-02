# from tokenizer import Tokenizer
import random
import math
import numpy as np
import pickle

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

embeddingMatrix = np.load("embeddingMatrixLookup.npy")  # Load the embedding matrix

# queen = embeddingMatrix[tokenizer.encode("king")] - embeddingMatrix[tokenizer.encode("man")]  + embeddingMatrix[tokenizer.encode("woman")]
# print("Queen Vector: ", queen)  # Vector for the word 'queen'
# print("Queen Vector Shape: ", queen.shape)  # Shape of the vector for the word
# # Find the most similar token vector in the embedding matrix
# token_id = np.argmax([cosine_similarity(queen, vec) for vec in embeddingMatrix])
# print("Most similar token to 'queen' (king - man + woman):", tokenizer.decode([token_id]))





# tokenizer = Tokenizer(vocabulary="low lowest John the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Wilson James John Robert Michael David William Richard Joseph Mary Patricia Jennifer Linda Elizabeth Barbara Susan Margaret\n Over hill, over dale, Thorough bush, thorough brier, Over park, over pale, Thorough flood, thorough fire! I do wander everywhere, Swifter than the moon's sphere; And I serve the Fairy Queen, To dew her orbs upon the green; The cowslips tall her pensioners be; In their gold coats spots you see; Those be rubies, fairy favours; In those freckles live their savours; I must go seek some dewdrops here, And hang a pearl in every cowslip's ear.", vocabSize=713)
# vocab_size = len(tokenizer.vocabulary)  
# embedding_dim = 50  # Can be 50, 100, 300, etc.

# # Randomly initialize
# embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim)) # Embedding matrix, used to find the word vector of the the input tokens
# print("\n\n\n\n\n\n")
# print("Vocabulary Size: ", vocab_size)
# print("Embedding Matrix Shape: ", embedding_matrix.shape)
# print("Embedding Matrix: \n", embedding_matrix)


# lowwjohn = tokenizer.encode("low w john is cool and amazing <PAD>")
# print("length of lowwjohn: ", len(lowwjohn))
# tokenWordVectors = embedding_matrix[lowwjohn]  # Get the word vectors for the encoded tokens which is the input of the transformer
# print("Token Word Vectors Shape: ", tokenWordVectors.shape)
# print("Token Word Vectors: \n", tokenWordVectors)
# print(" word vector of 'low': ", embedding_matrix[tokenizer.encode("low")])  # Example: vector for the first token 'low'
# print("Shape of the first token vector: ", embedding_matrix[tokenizer.encode("low")].shape) # Word Vector for 'low'
#print("Decoded: " + tokenizer.decode(lowwjohn))
# while len(lowwjohn) < 100:
#     lowwjohn.append(tokenizer.vocabulary["<PAD>"])  # Padding the input to a fixed length of 100 tokens


#  MAKE THE INPUT A FIXED LENGTH OF 10K TOKENS
class Transformer:
    def __init__(self, embeddingMatrix, learning_rate=0.00001):
        self.embeddingMatrix = embeddingMatrix  # Embedding matrix for the input tokens
        self.inputVector = np.zeros((500, 50))
        self.outputVector = np.zeros((500, 50))
        self.learning_rate = learning_rate
        self.w_Q = np.random.uniform(-0.1, 0.1, (50, 50)) # Query matrix
        self.w_K = np.random.uniform(-0.1, 0.1, (50, 50))# Key matrix
        self.w_V = np.random.uniform(-0.1, 0.1, (50, 50))  # Value matrix
        self.scores = []
        self.normalized_scores = []
        self.matrixLength = 0

    
    def forward(self, input_matrix):
        if len(input_matrix) > 500:
            raise ValueError("Input matrix exceeds the maximum length of 500 tokens.")
        self.matrixLength = len(input_matrix)
        input_matrix = [self.embeddingMatrix[token] for token in input_matrix]  # Convert tokens to their corresponding word vectors using the embedding matrix
        if len(input_matrix) < 500:
            padding_length = 500 - len(input_matrix)
            while len(input_matrix) < 500:
                input_matrix.append(self.embeddingMatrix[0])  # Padding the input to a fixed length of 100 tokens
        for o in range(self.matrixLength):
            for p in range(50):
                self.inputVector[o][p] = input_matrix[o][p]
        
        kScores = []
        vScores = []
        qScores = []
        for i in range(len(self.inputVector)):
            kScores.append(np.dot(self.inputVector[i], self.w_K))  # Transpose to match dimensions for dot product
            vScores.append(np.dot(self.inputVector[i], self.w_V))  # Transpose to match dimensions for dot product
            qScores.append(np.dot(self.inputVector[i], self.w_Q))  # Transpose to match dimensions for dot product


        self.scores = []
        self.normalized_scores = []
        for j in range(len(self.w_Q)):
            scores = []
            for k in range(self.matrixLength):
                score = np.dot(qScores[j], kScores[k]) / math.sqrt(len(self.w_Q))
                if np.isnan(score) or np.isinf(score):
                    score = 0.0
                scores.append(score)
            self.normalized_scores.append(self.softmax(scores))
        
        for l in range(len(self.w_V)):
            output = 0
            for m in range(self.matrixLength):
                output += self.normalized_scores[l][m] * vScores[m]
            self.outputVector[l] = output
        
        return self.outputVector[:self.matrixLength]
    
    def softmax(self, x):
            # e_x = np.exp(x - np.max(x))
            # return e_x / np.sum(e_x)
        x = np.array(x)
        x -= np.max(x)  # for numerical stability
        e_x = np.exp(x)
        return e_x / (np.sum(e_x) + 1e-9)
    
    def train(self, input_matrix, target_matrix, learning_rate=0.00001):
        if len(input_matrix) < len(target_matrix):
            raise ValueError("Input matrix length must be greater than or equal to target matrix length.")
        if len(input_matrix) > 500 or len(target_matrix) > 500:
            raise ValueError("Input and target matrices must not exceed the maximum length of 100 tokens.")
        
        self.learning_rate = learning_rate
        input_m = [self.embeddingMatrix[token] for token in input_matrix]  # Convert tokens to their corresponding word vectors using the embedding matrix
        target_v = [self.embeddingMatrix[token] for token in target_matrix]  # Convert tokens to their corresponding word vectors using the embedding matrix

        if len(input_m) < 500:
            padding_length = 500 - len(input_m)
            while len(input_m) < 500:
                input_m.append(self.embeddingMatrix[0])  # Padding the input to a fixed length of 100 tokens
        if len(target_v) < 500:
            padding_length = 500 - len(target_v)
            while len(target_v) < 500:
                target_v.append(self.embeddingMatrix[0])  # Padding the output to a fixed length of 100 tokens

    

        total_loss = 0
        for i in range(len(input_matrix)):
            input_vector = input_matrix[i]
            target_vector = target_v[i]
            # print("\n\nInput Vector: ", input_vector)
            # print("\n\nInput Vector: ",input_matrix)
                
            output_vector = self.forward([input_vector])
            # print("\n\nOutput Vector: ", output_vector)
            # Compute loss (mean squared error)
            loss = np.mean((output_vector - target_vector) ** 2)
            total_loss += loss
            
            # Backpropagation
            grad_output = 2 * (output_vector - target_vector) / len(output_vector)
            grad_output = np.clip(grad_output, -1.0, 1.0)
            grad_norm = np.linalg.norm(grad_output)
            if grad_norm > 1.0:
                grad_output = grad_output / grad_norm

            # print("\n\nGradient Output: ", grad_output[0])
            # print("\n\n weight Q: ", self.w_Q[0])
            
            # Update weights
            self.w_Q -= self.learning_rate * np.outer(grad_output, input_vector)
            self.w_K -= self.learning_rate * np.outer(grad_output, input_vector)
            self.w_V -= self.learning_rate * np.outer(grad_output, input_vector)

        # avg_loss = total_loss / len(input_matrix)
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return total_loss
        
      

# print("\n\n\nlowwjohn: ", lowwjohn)
# print("Lowwjohn: ", np.array(lowwjohn).shape)  # Shape of the padded input
# # inputWordVectors = embedding_matrix[lowwjohn] # Get the word vectors for the padded input tokens
# # print("Input Word Vectors Shape: ", inputWordVectors.shape)
# # print("Input Word Vectors: \n", inputWordVectors)

# transformer = Transformer(embedding_matrix)
# #  input tokens directly into the transformer without converting them into word vectors
# output = transformer.forward(lowwjohn)  # Forward pass through the transformer
# print("Output Vector Shape: ", output.shape)
# print("Output Vector: \n", output)
# decoded_output = []
# for i in range(len(output)):
#     decoded_output.append(tokenizer.decode([np.argmax(output[i])]))  # Decode each output vector back to text

# print("\nDecoded Output: ", decoded_output)  # Print the decoded output vectors






# TranModel = Transformer(embeddingMatrix)
# TranModel.train(tokenizer.encode("John is "), tokenizer.encode("cool"), epochs=2)  # Train the transformer model with the padded input and output

# output = TranModel.forward(tokenizer.encode("John "))  # Forward pass through the transformer
# print("Output Vector Shape: ", output.shape)
# print("Output Vector: \n", output)
# similarOutput = np.argmax(output, axis=1)  # Get the most similar token vector in the output
# print("\n\nDecoded Output: ", tokenizer.decode(similarOutput))

# output = TranModel.forward(tokenizer.encode(f"John {similarOutput}"))  # Forward pass through the transformer
# print("\nOutput Vector Shape: ", output.shape)
# print("Output Vector: \n", output)
# similarOutput = np.argmax(output, axis=1)  # Get the most similar token vector in the output
# print("\n\nDecoded Output: ", tokenizer.decode(similarOutput))