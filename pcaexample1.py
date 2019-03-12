from sklearn.datasets 
import load_iris iris = load_iris()
from sklearn.decomposition import PCA
# 2 for just retaining two components  
pca = PCA(2) 
#model specification
print(pca)

X, y = iris.data, iris.target 
X_proj = pca.fit_transform(X)
# retains 97 % of the variance in this case 
print X_proj.shape  
# shape becomes (150,2) 


