import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

ex_data = pd.read_csv('/Users/halim/Downloads/linkage example.csv')

ex_data.shape

ex_data.head()

# mergings = linkage(ex_data, method='complete')
#
# plt.title("Complete linkage")
# dendrogram(mergings)
# plt.show()
#
# mergings = linkage(ex_data, method='single')
# plt.title("Sinlge linkage")
# dendrogram(mergings)
# plt.show()
#
# mergings = linkage(ex_data, method='average')
# plt.title("Average linkage")
# dendrogram(mergings)
# plt.show()
#
# mergings = linkage(ex_data, method='centroid')
# plt.title("Centroid linkage")
# dendrogram(mergings)
# plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
print('<complete linkage>')
print(cluster.fit_predict(ex_data))

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
print('<single linkage>')
print(cluster.fit_predict(ex_data))

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
print('<Average linkage>')
print(cluster.fit_predict(ex_data))
