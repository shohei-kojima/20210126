#!/usr/bin/env python


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.linear_model import LinearRegression
matplotlib.rcParams['lines.linewidth']=0.5
matplotlib.rcParams['axes.linewidth']=0.5
matplotlib.rcParams['xtick.major.width']=0.5
matplotlib.rcParams['ytick.major.width']=0.5
matplotlib.rcParams['font.size']=5


# make test data
f='id_to_pop_crossref_1kGP.txt'
out=[]
with open(f) as infile:
    for line in infile:
        ls=line.split()
        if ls[2] == 'EUR':
            if np.random.rand() < 0.1:
                cpm=np.random.normal(150, 7.5, 1)[0]
            else:
                cpm=np.random.normal(100, 5, 1)[0]
        elif ls[2] == 'AFR':
            cpm=np.random.normal(200, 10, 1)[0]
        else:
            cpm=np.random.normal(100, 5, 1)[0]
        out.append('%s\t%s\t%f\n' % (ls[0], ls[2], cpm))
with open('CPM.txt', 'w') as outfile:
    outfile.write(''.join(out))


# load test data
df_cpm=pd.read_csv('CPM.txt', index_col=0, header=None, sep='\t', names=('ancestry', 'cpm'))


# plot test data
fig=plt.figure(figsize=(3, 3))  # (x, y)
ax=fig.add_subplot(111)
sns.swarmplot(data=df_cpm, x='ancestry', y='cpm', size=1)
plt.suptitle('CPM')
plt.savefig('plot_orig_cpm.pdf')
plt.close()


# load test data
df_cpm=pd.read_csv('CPM.txt', index_col=0, header=None, sep='\t', names=('ancestry', 'cpm'))

# load PC
df_pc=pd.read_csv('chr22_pca.eigenvec', index_col=0, sep='\t')

# join CPM and PC
df=df_cpm.join(df_pc, how='inner')

# convert data to zscore
X=st.zscore(df[df_pc.columns])
y= df['cpm'] - df['cpm'].mean()
y=st.zscore(y).reshape(-1, 1)


# linear regression
reg=LinearRegression().fit(X, y)


# calc. epsilon
eps= y.flatten() - (np.dot(X, reg.coef_.flatten()) + reg.intercept_)
eps=st.zscore(eps)
df_normalized=pd.DataFrame(eps, index=df.index, columns=['norm_cpm'])
df_normalized=df_normalized.join(df_cpm, how='inner')


# plot normalized value
fig=plt.figure(figsize=(3, 3))  # (x, y)
ax=fig.add_subplot(111)
sns.swarmplot(data=df_normalized, x='ancestry', y='norm_cpm', size=1)
plt.suptitle('Normalized CPM, zscore')
plt.savefig('plot_norm_cpm.pdf')
plt.close()


# check regression results
df_coef=pd.DataFrame(reg.coef_.flatten(), index=df_pc.columns, columns=['coef'])
print(df_coef)
print(reg.intercept_)

fig=plt.figure(figsize=(3, 3))  # (x, y)
ax=fig.add_subplot(111)
sns.scatterplot(data=df, x='PC1', y='PC2', hue='ancestry', s=3, alpha=0.3)
plt.suptitle('PC1 vs PC2')
plt.savefig('plot_PC1_PC2.pdf')
plt.close()

