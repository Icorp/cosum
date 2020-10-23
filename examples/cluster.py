from cosum.cosum import findK

text = "The rest of this paper is organized as follows. Section 2 introduces the overview of related work. In Section 3, mathematical formulation of sentence selection problem for text summarization is introduced. It first segregates the sentences into clusters by topics, and then, the sentence selection problem from each cluster is formulated as an optimization problem. Section 4 presents a modified DE algorithm for solving the optimization problem."
K = findK(4,text)
print(K)
