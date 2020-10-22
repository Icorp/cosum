# cosum
COSUM: Text summarization based on clustering and
optimization

Text summarization is a process of extracting salient information from a source text
and presenting that information to the user in a condensed form while preserving
its main content. In the text summarization, most of the difficult problems are providing wide topic coverage and diversity in a summary. Research based on clustering,
optimization, and evolutionary algorithm for text summarization has recently shown
good results, making this a promising area. In this paper, for a text summarization, a
two‐stage sentences selection model based on clustering and optimization techniques, called COSUM, is proposed.


# Stage One


Let `T = {t1, t2, …, tm}` represents all the distinct terms occurred in the document D, where m is the number of terms. Using the vector space model,
each sentence Si is represented as a vector in m‐dimensional space, `Si = [wi1, …, wim]`, i = 1, …, n, where each component reflects weight of a corresponding term. Different weighting schemes are available.

### Get tokens and compute weight

![Image of Yaktocat](images/weight_formula.png)



# Stage Two

After representation of sentences, now we can calculate the similarity between them. Intuitively, if there are many common words between
two sentences, they are very similar. Let given two sentences `Si = [wi1, …, wim]` and `Sj = [wj1, …, wjm]`. Then to measure similarity between them, we
use the following measure:

![Image of Yaktocat](images/similarity_formula.png)