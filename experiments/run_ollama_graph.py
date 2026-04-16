import ollama
import networkx as nx
import random
import matplotlib.pyplot as plt

topics = [
"how to cook rice",
"python programming tutorial",
"machine learning basics",
"how to lose weight",
"neural networks explanation",
"how to train a dog",
"data science statistics",
"deep learning architecture",
"healthy workout routine",
"how to bake bread"
]

G = nx.Graph()

ratios = []
sessions = []

for i in range(1,150):

    prompt = random.choice(topics)

    response = ollama.embeddings(
        model="mistral",
        prompt=prompt
    )

    vec = response["embedding"]

    # 取最大10個維度當 feature
    features = sorted(range(len(vec)), key=lambda x: vec[x])[-10:]

    for a in features:
        for b in features:
            if a != b:
                G.add_edge(a,b)

    if i % 10 == 0:

        largest = max(nx.connected_components(G), key=len)
        ratio = len(largest) / len(G.nodes)

        ratios.append(ratio)
        sessions.append(i)

        print(i, ratio)

plt.plot(sessions,ratios)
plt.xlabel("sessions")
plt.ylabel("|Cmax| / |V|")
plt.title("LLM semantic feature graph")
plt.show()
