"""
Find what's hot in your field based on the accepted papers in your favorite conference.
Reads a file with paper titles, possibly obtained from the conference web site.
Hint: I use Chrome's developer tools to format the conference web page to simplify 
the list and paste it into a txt file. It often takes a few minutes for me.

- Semih Yagcioglu
"""

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

file_name = 'cvpr2017-accepted-papers.txt'
line_mod = 1  # paper titles recur per how many lines
ngram_range = (2, 4)  # word range to consider
filter_top = 20  # show top K papers
keywords_to_look_for = ["comprehension", "reading", "question", "answer",
                        "reasoning", "comprehend"]

with open(file_name, 'r') as text_file:
    data = text_file.readlines()
    paper_list = [item.replace('\n', '').lower() for i, item in enumerate(data)
                  if i % line_mod == 0]

document = " ".join(paper_list)
ngrams = CountVectorizer(ngram_range=ngram_range, stop_words='english')
analyzer = ngrams.build_analyzer()
results = analyzer(document)

common_terms = Counter(results).most_common(filter_top)

print "\n\n\n"
print("There are {0} accepted papers".format(len(paper_list)))
print "-" * 50

for term, count in common_terms:
    print("{0:40}{1}".format(term, count))

interesting_papers = [paper for paper in paper_list if
                      any(keyword.lower() in paper for keyword in keywords_to_look_for)]

print "\n\n\n"
print("There are {0} interesting papers".format(len(interesting_papers)))
print "-" * 50

for interesting_paper in interesting_papers:
    print("{0:}".format(interesting_paper))
