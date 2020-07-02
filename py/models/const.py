data_prefix = "data"
interim_prefix = f"{data_prefix}/interim"
processed_prefix = f"{data_prefix}/processed"
raw_prefix = f"{data_prefix}/raw"
PATH = {
    "train": f"{raw_prefix}/train_data.csv",
    "test": f"{raw_prefix}/test_data.csv",
    "land_price": f"{raw_prefix}/published_land_price.csv",
    "train_intermediate": f"{interim_prefix}/train_data.jbl",
    "test_intermediate": f"{interim_prefix}/test_data.jbl",
    "land_price_intermediate": f"{interim_prefix}/published_land_price.jbl",
    "X_train_processed": f"{processed_prefix}/X_train_.jbl",
    "y_train_processed": f"{processed_prefix}/y_train_.jbl",
    "X_test_processed": f"{processed_prefix}/X_test_.jbl",
    "prefix": {
        "processed": f"{processed_prefix}",
        "logs": "logs",
        "adversarial": "models/adversarial",
        "importance": "models/importance",
        "model": "models/model",
        "optuna": "models/optuna",
        "prediction": "models/prediction",
        "submission": "submission",
    },
}

COLUMN = {
    "categorical": [
        "Type",
        "Region",
        # "MunicipalityCode",
        # "Prefecture",
        "Municipality",
        "DistrictName",
        "NearestStation",
        "FloorPlan",
        "LandShape",
        "Frontage",
        "Structure",
        "Use",
        "Purpose",
        "Direction",
        "Classification",
        "CityPlanning",
        "Renovation",
    ],
    "drop": ["id", "Prefecture", "BuildingYear", "Period", "Remarks"],
    "target": "y",
}
