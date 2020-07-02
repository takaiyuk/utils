from typing import Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import log_artifact, log_metric, log_param
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from src.const import PATH
from src.data.load import Loader
from src.models import ModelLGBM, ModelOptunaLGBM
from src.models.model import Model
from src.utils.joblib import Jbl
from src.utils.logger import Logger

logger = Logger()
models_map = {
    "ModelLGBM": ModelLGBM,
    "ModelOptunaLGBM": ModelOptunaLGBM,
}


class Runner:
    """
    example
    -------
    > run_configs = load_yaml(args.run)
    > cv = StratifiedGroupKFold(
    >     n_splits=run_configs["kfold"]["number"], random_state=run_configs["seed"]
    > )

    > runner = Runner(run_configs, cv)
    > runner.run_train_cv()
    > runner.run_predict_cv()
    > runner.submission()
    """
    def __init__(self, config: dict, cv):
        self.exp_name = config["exp_name"]
        self.run_name = config["run_name"]
        self.run_id = None
        self.fe_name = config["fe_name"]
        self.X_train = Jbl.load(
            f"{PATH['prefix']['processed']}/X_train_{config['fe_name']}.jbl"
        )
        self.y_train = Jbl.load(
            f"{PATH['prefix']['processed']}/y_train_{config['fe_name']}.jbl"
        )
        self.X_test = Jbl.load(
            f"{PATH['prefix']['processed']}/X_test_{config['fe_name']}.jbl"
        )
        self.evaluation_metric = config["evaluation_metric"]
        self.params = config["params"]
        self.cols_definition = config["cols_definition"]
        self.kfold = config["kfold"]["method"]
        self.cv = cv
        self.description = config["description"]
        self.advanced = config["advanced"] if "advanced" in config else None

        if config["model_name"] in models_map.keys():
            self.model_cls = models_map[config["model_name"]]
        else:
            raise ValueError

    def train_fold(self, i_fold: int):
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        # 残差の設定
        if self.advanced and "ResRunner" in self.advanced:
            oof = Jbl.load(self.advanced["ResRunner"]["oof"])
            X_train["res"] = (y_train - oof).abs()

        # 学習データ・バリデーションデータをセットする
        tr_idx, va_idx = self.load_index_fold(i_fold)
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # 残差でダウンサンプリング
        if self.advanced and "ResRunner" in self.advanced:
            X_tr = X_tr.loc[
                (X_tr["res"] < self.advanced["ResRunner"]["res_threshold"]).values
            ]
            y_tr = y_tr.loc[
                (X_tr["res"] < self.advanced["ResRunner"]["res_threshold"]).values
            ]
            print(X_tr.shape)
            X_tr.drop("res", axis=1, inplace=True)
            X_val.drop("res", axis=1, inplace=True)

        # Pseudo Lebeling
        if self.advanced and "PseudoRunner" in self.advanced:
            y_test_pred = Jbl.load(self.advanced["PseudoRunner"]["y_test_pred"])
            if "pl_threshold" in self.advanced["PseudoRunner"]:
                X_add = self.X_test.loc[
                    (y_test_pred < self.advanced["PseudoRunner"]["pl_threshold"])
                    | (y_test_pred > 1 - self.advanced["PseudoRunner"]["pl_threshold"])
                ]
                y_add = pd.DataFrame(y_test_pred).loc[
                    (y_test_pred < self.advanced["PseudoRunner"]["pl_threshold"])
                    | (y_test_pred > 1 - self.advanced["PseudoRunner"]["pl_threshold"])
                ]
                y_add = pd.DataFrame(([1 if ya > 0.5 else 0 for ya in y_add[0]]))
            elif "pl_threshold_neg" in self.advanced["PseudoRunner"]:
                X_add = self.X_test.loc[
                    (y_test_pred < self.advanced["PseudoRunner"]["pl_threshold_neg"])
                    | (y_test_pred > self.advanced["PseudoRunner"]["pl_threshold_pos"])
                ]
                y_add = pd.DataFrame(y_test_pred).loc[
                    (y_test_pred < self.advanced["PseudoRunner"]["pl_threshold_neg"])
                    | (y_test_pred > self.advanced["PseudoRunner"]["pl_threshold_pos"])
                ]
                y_add = pd.DataFrame(([1 if ya > 0.5 else 0 for ya in y_add[0]]))
            else:
                X_add = self.X_test
                y_add = pd.DataFrame(y_test_pred)
            print(f"added X_test: {len(X_add)}")
            X_tr = pd.concat([X_tr, X_add])
            y_tr = pd.concat([y_tr, y_add])

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(X_tr, y_tr, X_val, y_val, self.X_test)

        # バリデーションデータへの予測・評価を行う
        pred_val = model.predict(X_val)

        # 後処理
        pred_val = postprocess(pred_val)

        score = self.evaluate(y_val.values, pred_val)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, pred_val, score

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # mlflow
        mlflow.set_experiment(self.exp_name)
        mlflow.start_run(run_name=self.run_name)
        logger.info(f"{self.run_name} - start training cv")

        scores = []
        va_idxes = []
        preds = []

        # Adversarial validation
        if self.advanced and "adversarial_validation" in self.advanced:
            X_train = self.X_train
            X_test = self.X_test
            X_train["target"] = 0
            X_test["target"] = 1
            X_train = pd.concat([X_train, X_test], sort=False).reset_index(drop=True)
            y_train = X_train["target"]
            X_train.drop("target", axis=1, inplace=True)
            X_test.drop("target", axis=1, inplace=True)
            self.X_train = X_train
            self.y_train = y_train

        # 各foldで学習を行う
        for i_fold in range(self.cv.n_splits):
            # 学習を行う
            logger.info(f"{self.run_name} fold {i_fold} - start training")
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(
                f"{self.run_name} fold {i_fold} - end training - score {score}\tbest_iteration: {model.model.best_iteration}"
            )

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        cv_score = self.evaluate(self.y_train.values, preds)

        logger.info(
            f"{self.run_name} - end training cv - score {cv_score}\tbest_iteration: {model.model.best_iteration}"
        )

        # 予測結果の保存
        Jbl.save(preds, f"{PATH['prefix']['prediction']}/{self.run_name}-train.jbl")

        # mlflow
        self.run_id = mlflow.active_run().info.run_id
        log_param("model_name", self.model_cls.__class__.__name__)
        log_param("fe_name", self.fe_name)
        log_param("train_params", self.params)
        log_param("cv_strategy", str(self.cv))
        log_param("evaluation_metric", self.evaluation_metric)
        log_metric("cv_score", cv_score)
        log_param(
            "fold_scores",
            dict(
                zip(
                    [f"fold_{i}" for i in range(len(scores))],
                    [round(s, 4) for s in scores],
                )
            ),
        )
        log_param("cols_definition", self.cols_definition)
        log_param("description", self.description)
        mlflow.end_run()

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """

        logger.info(f"{self.run_name} - start prediction cv")
        X_test = self.X_test
        preds = []

        show_feature_importance = "LGBM" in str(self.model_cls)
        if show_feature_importance:
            feature_importances = pd.DataFrame()

        # 各foldのモデルで予測を行う
        for i_fold in range(self.cv.n_splits):
            logger.info(f"{self.run_name} - start prediction fold:{i_fold}")
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(X_test)
            preds.append(pred)
            logger.info(f"{self.run_name} - end prediction fold:{i_fold}")
            if show_feature_importance:
                feature_importances = pd.concat(
                    [feature_importances, model.feature_importance(X_test)], axis=0
                )

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Jbl.save(pred_avg, f"{PATH['prefix']['prediction']}/{self.run_name}-test.jbl")

        logger.info(f"{self.run_name} - end prediction cv")

        # 特徴量の重要度
        if show_feature_importance:
            aggs = (
                feature_importances.groupby("Feature")
                .mean()
                .sort_values(by="importance", ascending=False)
            )
            cols = aggs[:200].index
            pd.DataFrame(aggs.index).to_csv(
                f"{PATH['prefix']['importance']}/{self.run_name}-fi.csv", index=False
            )

            best_features = feature_importances.loc[
                feature_importances.Feature.isin(cols)
            ]
            plt.figure(figsize=(14, 26))
            sns.barplot(
                x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance", ascending=False),
            )
            plt.title("LightGBM Features (averaged over folds)")
            plt.tight_layout()
            plt.savefig(f"{PATH['prefix']['importance']}/{self.run_name}-fi.png")
            plt.show()

            # mlflow
            mlflow.start_run(run_id=self.run_id)
            log_artifact(f"{PATH['prefix']['importance']}/{self.run_name}-fi.png")
            mlflow.end_run()

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(
            run_fold_name, self.params, self.cols_definition["categorical_col"]
        )

    def load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        if self.kfold == "normal":
            return list(self.cv.split(self.X_train, self.y_train))[i_fold]
        elif self.kfold == "stratified":
            return list(self.cv.split(self.X_train, self.y_train))[i_fold]
        elif self.kfold == "group":
            return list(
                self.cv.split(
                    self.X_train,
                    self.y_train,
                    groups=self.X_train[self.cols_definition["cv_group"]],
                )
            )[i_fold]
        elif self.kfold == "stratified_group":
            return list(
                self.cv.split(
                    self.X_train,
                    self.y_train,
                    groups=self.X_train[self.cols_definition["cv_group"]],
                )
            )[i_fold]
        else:
            raise Exception("Invalid kfold method")

    def evaluate(self, y_true: np.array, y_pred: np.array) -> float:
        """指定の評価指標をもとにしたスコアを計算して返す
        :param y_true: 真値
        :param y_pred: 予測値
        :return: スコア
        """
        if self.evaluation_metric == "log_loss":
            score = log_loss(y_true, y_pred, eps=1e-15, normalize=True)
        elif self.evaluation_metric == "mean_absolute_error":
            score = mean_absolute_error(y_true, y_pred)
        elif self.evaluation_metric == "rmse":
            score = np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.evaluation_metric == "rmsle":
            score = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
        elif self.evaluation_metric == "auc":
            score = roc_auc_score(y_true, y_pred)
        elif self.evaluation_metric == "prauc":
            score = average_precision_score(y_true, y_pred)
        else:
            raise Exception("Unknown evaluation metric")
        return score

    def submission(self):
        pred = Jbl.load(f"{PATH['prefix']['prediction']}/{self.run_name}-test.jbl")
        sub = Loader().load_test().loc[:, ["id"]]
        if self.advanced and "predict_exp" in self.advanced:
            sub[self.cols_definition["target_col"]] = np.exp(pred)
        else:
            sub[self.cols_definition["target_col"]] = pred
        sub.to_csv(
            f"{PATH['prefix']['submission']}/submission_{self.run_name}.csv",
            index=False,
        )

    def reset_mlflow(self):
        mlflow.end_run()


def postprocess(pred: np.array) -> np.array:
    return pred
