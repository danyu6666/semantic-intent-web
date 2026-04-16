import random
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

ratios = []
sessions = []

for i in range(1,2000):

    features = random.sample(range(500), 3)

    for a in features:
        for b in features:
            if a != b:
                G.add_edge(a,b)

    if i % 50 == 0:

        largest = max(nx.connected_components(G), key=len)

        ratio = len(largest)/len(G.nodes)

        ratios.append(ratio)
        sessions.append(i)

        print(i, ratio)

plt.plot(sessions,ratios)
plt.xlabel("sessions")
plt.ylabel("|Cmax|/|V|")
plt.title("test percolation")
plt.show()
