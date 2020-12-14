from paragraph_classifier import ParagraphClassifier
a = ParagraphClassifier(seq_maxlen=300)
a._load_embeddings()
a.load()

text = open('2.txt', 'r')
res = []
for content in text:
    res.append(content)
#print(res) # cut text contents into paragraphs and stored in res (type: list)

paragraph_position = []
for i, element in enumerate(res):
    paragraph_position.append([i])

paragraph_text = res[24]
paragraph_position1 = paragraph_position[24]

prediction = a.predict_one(paragraph_text, paragraph_position1)
print(prediction)
