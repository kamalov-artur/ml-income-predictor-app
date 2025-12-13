import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


df_dirty = pd.read_csv('../data/adult.csv')
df_dirty.columns[(df_dirty == ('?')).any()]
df_dirty.columns = (x.replace('-', '_') for x in df_dirty.columns)
df = df_dirty.replace('?', np.nan).dropna()


y = (df['>50K,<=50K'] == '>50K').astype(int)
X = df.drop(columns=['>50K,<=50K'])

cat_features = X.select_dtypes(include=['object']).columns

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

model = GradientBoostingClassifier(
    n_estimators=300,
    criterion='squared_error',
    max_features='log2',
    random_state=20251210
)

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

pipeline.fit(X, y)

joblib.dump(pipeline, 'model.pkl')
