import pandas as pd
import numpy as np
from tabulate import tabulate
from itertools import combinations

b1_dir = '../instacart/products.csv/'

products = pd.read_csv(b1_dir + 'products.csv')
dic = products.set_index('product_id')['product_name'].to_dict()

pregnancy_user_product = pd.read_csv('pregnancy_user_product.csv')
pup = pregnancy_user_product


matrix0 = pd.pivot_table(pup, index='user_id',
                         columns='product_id', values='reordered', aggfunc=len, fill_value=0)

x_cols = [col for col in matrix0.columns if col in dic]
matrix = matrix0.loc[:, x_cols]


#matrix = matrix.applymap(lambda x: 1 if x > 0 else 0)
#matrix = matrix.fillna(0).reset_index()


#from sklearn.cluster import KMeans
#cluster = KMeans(n_clusters=5)

import hdbscan
cluster = hdbscan.HDBSCAN(min_cluster_size=10)
# slice matrix so we only include the 0/1 indicator columns in the clustering
matrix['cluster'] = cluster.fit_predict(matrix[x_cols])
matrix.cluster.value_counts()

from plotnine import *
ggplot(matrix, aes(x='factor(cluster)')) + geom_bar() + \
    xlab("Cluster") + ylab("Customers\n(# in cluster)")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix2d = pca.fit_transform(matrix[x_cols])
matrix['x'] = matrix2d[:, 0]
matrix['y'] = matrix2d[:, 1]
#matrix = matrix.reset_index()

customer_clusters = matrix[['cluster', 'x', 'y']]
customer_clusters.head()

p = ggplot(customer_clusters, aes(x='x', y='y', color='factor(cluster)')) + \
    geom_point(size=3) + \
    ggtitle("Customers Grouped by Cluster")

ggsave(plot=p, filename='customer_cluster.png')

cluster0 = matrix[matrix.cluster == 11]
for user_id, row in cluster0.iterrows():
    print(user_id, [dic[col] for col in x_cols if row[col] > 0])

for product_id in cluster0[x_cols].sum().reset_index().sort_values(0, ascending=False)['product_id'].tolist()[:50]:
    print(dic[product_id])

print('-----')

for product_id in cluster0[x_cols].applymap(lambda x: 1 if x > 0 else 0).sum().reset_index().sort_values(0, ascending=False)['product_id'].tolist()[:50]:
    print(dic[product_id])

# https://stackoverflow.com/questions/20209600/panda-dataframe-remove-constant-column


def c(df):
    return df.loc[:, (df != df.iloc[0]).any()]


cluster0_clean = cluster0.loc[:, [
    col for col in cluster0.columns if col in x_cols]]
cluster0_clean = c(cluster0_clean)
cluster0_clean.columns = [dic.get(col, col) for col in cluster0_clean.columns]
import seaborn as sns
from tastu_teche.plt_show import plt_show, df_show, set_show, plt_figure, ax_hbar_value, ax_vbar_value
sns.set()
import matplotlib.pyplot as plt
plt_figure(16)
ax = sns.heatmap(cluster0_clean.T, annot=False, linewidths=.5)
ax.set_title('user_id x product(quantity bought)')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt_show('user_id_product.png')
