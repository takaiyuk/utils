import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd

from .file import mkdir
from .joblib import Jbl


class BaseModel(metaclass=ABCMeta):
    """https://github.com/upura/ayniy/blob/master/ayniy/model/model.py"""

    def __init__(
        self,
        params: Dict[str, Any],
        categorical_features: Optional[List[str]] = None,
    ) -> None:
        self.model: Any = None
        self.params = params
        self.categorical_features = categorical_features
        _class_name = self.__class__.__name__
        if _class_name not in [
            "CatClassifierModel",
            "CatRegressorModel",
            "LGBMModel",
            "LGBMOptunaModel",
        ]:
            print(f"[WARNING]{_class_name} cannot handle categorical features")

    @abstractmethod
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        :param X_tr: 学習データの特徴量
        :param y_tr: 学習データの目的変数
        :param X_val: バリデーションデータの特徴量
        :param y_val: バリデーションデータの目的変数
        """
        pass

    def predict(self, X_te: pd.DataFrame) -> np.ndarray:
        """学習済のモデルでの予測値を返す
        :param X_te: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        if self.model is None:
            raise ValueError("train model before predict")
        return self.model.predict(X_te)

    def save_model(self, model_path: str) -> None:
        """モデルの保存を行う
        :param path: モデルの保存先パス
        """
        model_path_dir = os.path.dirname(model_path)
        mkdir(model_path_dir)
        Jbl.save(self.model, model_path)

    def load_model(self, model_path: str) -> None:
        """モデルの読み込みを行う
        :param path: モデルの読み込み先パス
        """
        self.model = Jbl.load(model_path)

    def save_params(self, path: str) -> None:
        """path: str = f'{OutputPath.optuna}/best_params.json'"""
        with open(path, "w") as f:
            json.dump(self.model.params, f, indent=4, separators=(",", ": "))


class LGBMModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        # データのセット
        is_validation = X_val is not None
        lgb_train = lgb.Dataset(
            X_tr, y_tr, categorical_feature=self.categorical_features
        )
        if is_validation:
            lgb_eval = lgb.Dataset(
                X_val,
                y_val,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
            )

        # ハイパーパラメータの設定
        params = self.params.copy()
        if "num_boost_round" in params.keys():
            num_round = params.pop("num_boost_round")
        elif "n_estimators" in params.keys():
            num_round = params.pop("n_estimators")
        else:
            print(
                "[WARNING] num_round is set to 100: `num_boost_round` or `n_estimators` are not in the params"
            )
            num_round = 100

        # 学習
        if is_validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=500,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.model = lgb.train(
                params, lgb_train, num_round, valid_sets=[lgb_train], verbose_eval=500
            )

    def feature_importance(
        self, fold_i: Optional[int] = None, importance_type: str = "gain"
    ) -> pd.DataFrame:
        df_fi = pd.DataFrame()
        df_fi["fold"] = fold_i
        df_fi["feature"] = self.model.feature_name()
        df_fi["importance"] = self.model.feature_importance(
            importance_type=importance_type
        )
        return df_fi


class LGBMOptunaModel(LGBMModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        # データのセット
        is_validation = X_val is not None
        lgb_train = optuna_lgb.Dataset(
            X_tr,
            y_tr,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        if is_validation:
            lgb_eval = optuna_lgb.Dataset(
                X_val,
                y_val,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
                free_raw_data=False,
            )

        # ハイパーパラメータの設定
        params = self.params.copy()
        if "num_boost_round" in params.keys():
            num_round = params.pop("num_boost_round")
        elif "n_estimators" in params.keys():
            num_round = params.pop("n_estimators")
        else:
            print(
                "[WARNING] num_round is set to 100: `num_boost_round` or `n_estimators` are not in the params"
            )
            num_round = 100

        # 学習
        if is_validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = optuna_lgb.train(  # type: ignore
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=1000,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.model = optuna_lgb.train(  # type: ignore
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train],
                verbose_eval=1000,
            )
