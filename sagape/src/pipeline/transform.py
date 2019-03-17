from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition.pca import PCA

def get_standard_scale_with_pca_etl() -> Pipeline :
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA())
    ])

    return pipe