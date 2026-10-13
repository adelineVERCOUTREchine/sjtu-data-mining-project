# ======== runMinHashExample =======
# This example code demonstrates comparing documents using the MinHash
# approach.
#
# First, each document is represented by the set of shingles it contains. The
# documents can then be compared using the Jaccard similarity of their
# shingle sets. This is computationally expensive, however, for large numbers
# of documents.
#
# For comparison, we will use the MinHash algorithm to calculate short
# signature vectors to represent the documents. These MinHash signatures can
# then be compared quickly by counting the number of components in which the
# signatures agree. We'll compare all possible pairs of documents, and find
# the pairs with high similarity.
#


import sys
import random
import time
from hashlib import sha1

# This is the number of components in the resulting MinHash signatures.
# Correspondingly, it is also the number of random hash functions that
# we will need in order to calculate the MinHash.
numHashes = 10

# Read data
numDocs = 4985
dataFile = "../archive/anime_synopsis.train"

# =============================================================================
#               Convert Documents To Sets of Shingles
# =============================================================================

print("Shingling articles...")

# Create a dictionary of the articles, mapping the article identifier (e.g.,
# "t8470") to the list of shingle IDs that appear in the document.
docsAsShingleSets = {}

# Open the data file.
f = open(dataFile, "r", encoding="utf-8")

docNames = []

t0 = time.time()

totalShingles = 0

for i in range(0, numDocs):

    # Read all of the words (they are all on one line) and split them by white
    # space.
    words = f.readline().rstrip("\r\n").split(" ")

    # Retrieve the article ID, which is the first word on the line.
    docID = words[0]

    # Maintain a list of all document IDs.
    docNames.append(docID)

    del words[0]

    # 'shinglesInDoc' will hold all of the unique shingle IDs present in the
    # current document. If a shingle ID occurs multiple times in the document,
    # it will only appear once in the set (this is a property of Python sets).
    shinglesInDoc = set()

    ######## TODO ########
    # Convert the article into a shingle set
    # Each shingle should contain 3 tokens
    # You should use sha1 (imported from hashlib at the beginning of this file)
    # as the hash function that maps each shingle into a 32-bit integer.
    # You may use int.from_bytes and hashlib._HASH.digest methods.
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i + 3])
        shingle_hash = int.from_bytes(sha1(shingle.encode('utf-8')).digest()[:4], 'big')
        shinglesInDoc.add(shingle_hash)

    ##### end of TODO #####

    # Store the completed list of shingles for this document in the dictionary.
    docsAsShingleSets[docID] = shinglesInDoc

    # Count the number of shingles across all documents.
    totalShingles = totalShingles + (len(words) - 2)

# Close the data file.
f.close()

# Report how long shingling took.
print("\nShingling " + str(numDocs) + " docs took %.2f sec." % (time.time() - t0))

print("\nAverage shingles per doc: %.2f" % (totalShingles / numDocs))

# =============================================================================
#            Define Matrices to Store extimated JSim
# =============================================================================

### define a table `estJSim` to store estimated Jsim  ###
### Hint for efficiency:
### http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
### The Triangular-Matrix Method section

estJSim = [None for _ in range(numDocs * (numDocs - 1) // 2)]


def getTriangleIndex(i, j) -> int:
    if i < j:
        return int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1
    elif i > j:
        return getTriangleIndex(j, i)
    else:
        raise ValueError("Indices must be different")


# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================

# Time this step.
t0 = time.time()

print("\nGenerating random hash functions...")

# Record the maximum shingle ID that we assigned.
maxShingleID = 2**32 - 1

# We need the next largest prime number above 'maxShingleID'.
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID, i.e. nextPrime
# For convenience, I have randomly picked them and fix them for you.
a_list = [
    2647644122,
    3144724950,
    1813742201,
    3889397089,
    850218610,
    4228854938,
    3422847010,
    1974054350,
    1398857723,
    3861451095,
]
b_list = [
    2834859619,
    3834190079,
    3272971987,
    1421011856,
    1598897977,
    1288507477,
    1224561085,
    3278591730,
    1664131571,
    3749293552,
]
# So the i-th hash function is
# h(x) = (a_list[i] * x + b_list[i]) % nextPrime

print("\nGenerating MinHash signatures for all documents...")

# List of documents represented as signature vectors
signatures = []

# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.

# For each document...
for docID in docNames:

    # Get the shingle set for this document.
    shingleIDSet = docsAsShingleSets[docID]

    # The resulting minhash signature for this document. It is supposed to be a list of ints.
    signature = []

    ###### TODO ######
    # complete the signature of this doc
    # i.e. set the list `signature' to a proper value.
    for k in range(len(a_list)):
        min_hash = nextPrime + 1  # Initialize to a value larger than any possible hash
        for shingle_id in shingleIDSet:
            # Compute the hash value for this shingle using the k-th hash function
            hash_value = (a_list[k] * shingle_id + b_list[k]) % nextPrime
            # Update the minimum hash value if this one is smaller
            if hash_value < min_hash:
                min_hash = hash_value
        signature.append(min_hash)

    ### end of TODO ###
    # Store the MinHash signature for this document.
    signatures.append(signature)

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0

print("\nGenerating MinHash signatures took %.2fsec" % elapsed)

# =============================================================================
#                     Compare All Signatures
# =============================================================================

print("\nComparing all signatures...")

# Creates a N x N matrix initialized to 0.

# Time this step.
t0 = time.time()

# For each of the test documents...
for i in range(0, numDocs):
    # Get the MinHash signature for document i.
    signature1 = signatures[i]

    # For each of the other test documents...
    for j in range(i + 1, numDocs):

        # Get the MinHash signature for document j.
        signature2 = signatures[j]

        ####### TODO #######
        # calculate the estimated Jaccard Similarity
        # Then store the value into estJSim
        JSim = 0.0
        for k in range(numHashes):
            if signature1[k] == signature2[k]:
                JSim += 1
        JSim /= numHashes
        estJSim[getTriangleIndex(i, j)] = JSim


        ##### end of TODO #####

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0

print("\nComparing MinHash signatures took %.2fsec" % elapsed)


# =============================================================================
#                   Display Similar Document Pairs
# =============================================================================

threshold = 0.1
print("\nList of Document Pairs with J(d1,d2) more than", threshold)
print("Values shown are the estimated Jaccard similarity and the actual")
print("Jaccard similarity.\n")
print("                   Est. J   Act. J")

# For each of the document pairs...
f = open("../data/prediction.csv", "w", encoding="utf-8")
f.write("article1,article2,Est. J,Act. J\n")
for i in range(0, numDocs):
    for j in range(i + 1, numDocs):

        estJ = estJSim[getTriangleIndex(i, j)]

        # If the similarity is above the threshold...
        if estJ > threshold:

            ###### TODO ######
            # Calculate the actual Jaccard similarity between two docs (shingle sets) for validation.
            
            J=0
            for shingle in docsAsShingleSets[docNames[i]]:
                if shingle in docsAsShingleSets[docNames[j]]:
                    J += 1
            J /= (len(docsAsShingleSets[docNames[i]]) + len(docsAsShingleSets[docNames[j]]) - J)  # You should set the actual Jaccard similarity here.

            ### end of TODO ###

            # Print out the match and similarity values with pretty spacing.
            print("  %5s --> %5s   %.2f     %.2f" % (docNames[i], docNames[j], estJ, J))
            f.write("{},{},{},{}\n".format(docNames[i], docNames[j], estJ, J))

f.close()