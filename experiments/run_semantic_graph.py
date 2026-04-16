from sentence_transformers import SentenceTransformer
import networkx as nx
import random
import matplotlib.pyplot as plt

model = SentenceTransformer("all-MiniLM-L6-v2")

prompts = [
"how to cook rice",
"how to bake bread",
"python programming tutorial",
"machine learning basics",
"neural networks explanation",
"best way to train a dog",
"how to lose weight",
"workout routine for beginners",
"deep learning architecture",
"statistics for data science"
]

G = nx.Graph()

ratios = []
sessions = []

for i in range(1,200):

    text = random.choice(prompts)

    embedding = model.encode(text)

    features = embedding.argsort()[-10:]

    for a in features:
        for b in features:
            if a != b:
                G.add_edge(int(a),int(b))

    if i % 10 == 0:

        largest = max(nx.connected_components(G), key=len)

        ratio = len(largest)/len(G.nodes)

        ratios.append(ratio)
        sessions.append(i)

        print(i,ratio)

plt.plot(sessions,ratios)
plt.xlabel("sessions")
plt.ylabel("|Cmax| / |V|")
plt.title("semantic feature graph")
plt.show()
