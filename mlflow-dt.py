
from pprint import pprint
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from urllib.parse import urlparse
import mlflow
import mlflow.xgboost


    
    # prepare example dataset
    

df = pd.read_csv('surv-analysis.csv')
X = df.drop('survival_percentage',axis = 1)
Y = df["survival_percentage"]

X_train ,X_test , y_train , y_test = train_test_split(X,Y , test_size=0.33,random_state=1)

with mlflow.start_run():
    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()

    xgbmodel = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3,
              enable_categorical=False, gamma=0.4, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=1, monotone_constraints='()',
              n_estimators=100, n_jobs=1, nthread=1, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0,
              silent=True, subsample=1, tree_method='exact')
    
    xgbmodel.fit(X_train, y_train)
    y_pred = xgbmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

   # run_id = mlflow.last_active_run().info.run_id
   # print("Logged data and model in run {}".format(run_id))
    if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(xgbmodel, "model", registered_model_name="surv-anl-mlflow")
    else:
            mlflow.sklearn.log_model(xgbmodel, "model")
