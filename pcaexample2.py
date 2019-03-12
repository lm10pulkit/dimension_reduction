from sklearn.datasets import load_digits 
digits = load_digits() 
print digits.keys() 
# this retains 29 components and 95% variance
X,y = digits.data, digits.target 
pca_digits=PCA(0.95) 
X_proj = pca_digits.fit_transform(X) 
print X.shape, X_proj.shape 

# now with only 2 components and retains 29 percent 
pca_digits=PCA(2) 
X_proj = pca_digits.fit_transform(X) 
print np.sum(pca_digits.explained_variance_ratio_) 