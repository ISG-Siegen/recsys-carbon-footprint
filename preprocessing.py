import importlib
import json
import re
from collections import Counter
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path

data_sets = ['Amazon2018-Books', 'Amazon2018-CDs-And-Vinyl', 'Amazon2018-Clothing-Shoes-And-Jewelry',
             'Amazon2018-Electronics', 'Amazon2018-Sports-And-Outdoors', 'Amazon2018-Toys-And-Games', 'Gowalla',
             'Hetrec-LastFM', 'MovieLens-100K', 'MovieLens-100K-Ratings', 'MovieLens-1M', 'MovieLens-1M-Ratings',
             'MovieLens-Latest-Small', 'Netflix', 'Retailrocket', 'Yelp-2018']

for data_set_name in data_sets:
    print(f"Processing {data_set_name}...")

    # set some base paths
    data_set_folder_base_path = Path(f"./data_sets/{data_set_name}")
    source_files_path = Path(f"./data_sets/{data_set_name}/source/files")
    link_file_path = Path(f"./data_sets/{data_set_name}/source/site/link.txt")

    processed_folder_path = Path(f"./data_sets/{data_set_name}/processed")
    processed_data_path = Path(f"./data_sets/{data_set_name}/processed/interactions.csv")
    processed_data_log_path = Path(f"./data_sets/{data_set_name}/processed/processing_log.txt")
    processed_data_metadata_path = Path(f"./data_sets/{data_set_name}/processed/metadata.json")

    atomic_folder_path = Path(f"./data_sets/{data_set_name}/atomic")
    atomic_data_path = Path(f"./data_sets/{data_set_name}/atomic/{data_set_name}.inter")
    atomic_data_log_path = Path(f"./data_sets/{data_set_name}/atomic/processing_log.txt")
    atomic_data_metadata_path = Path(f"./data_sets/{data_set_name}/atomic/metadata.json")

    data = None
    meta_data = None
    data_splits = None
    feedback_type = None

    user_column_name = "user"
    item_column_name = "item"
    rating_column_name = "rating"
    timestamp_column_name = "timestamp"

    # generate processed data path
    Path.mkdir(Path(processed_folder_path), exist_ok=True)

    # get the data loader for the current data set
    if data_set_name == "MovieLens-100K-Ratings":
        data_loader_name = "MovieLens-100K"
    elif data_set_name == "MovieLens-1M-Ratings":
        data_loader_name = "MovieLens-1M"
    else:
        data_loader_name = data_set_name
    class_name = re.sub(r'[^a-zA-Z0-9]', '', data_loader_name)
    module_name = f"data_loaders.{re.sub(r'[^a-zA-Z0-9]', '_', data_loader_name).lower()}"

    # load data
    try:
        loader = importlib.import_module(module_name).__getattribute__(class_name)
        data = loader.load_from_file(source_path=source_files_path,
                                     user_column_name=user_column_name,
                                     item_column_name=item_column_name,
                                     rating_column_name=rating_column_name,
                                     timestamp_column_name=timestamp_column_name)
    except ModuleNotFoundError:
        print(f"Data loader for {data_set_name} not found.")
        pass

    # drop duplicate user-item interactions
    data.drop_duplicates(subset=[user_column_name, item_column_name], keep="last", inplace=True)

    # normalize identifiers
    for col in [user_column_name, item_column_name]:
        unique_ids = {key: value for value, key in enumerate(data[col].unique())}
        data[col].update(data[col].map(unique_ids))

    # order columns
    columns = list(data)
    num_columns = len(columns)
    if num_columns == 2:
        data = data[[user_column_name, item_column_name]]
    elif num_columns == 3:
        if rating_column_name in columns:
            data = data[[user_column_name, item_column_name, rating_column_name]]
        elif timestamp_column_name in columns:
            data = data[[user_column_name, item_column_name, timestamp_column_name]]
    elif num_columns == 4:
        data = data[[user_column_name, item_column_name, rating_column_name, timestamp_column_name]]
    else:
        print(f"Data set {data_set_name} has an unexpected number of columns.")
        pass

    # set data types
    data[user_column_name] = data[user_column_name].astype(str)
    data[item_column_name] = data[item_column_name].astype(str)
    if timestamp_column_name in list(data):
        data[timestamp_column_name] = data[timestamp_column_name].astype(str)
    if rating_column_name in list(data):
        data[rating_column_name] = data[rating_column_name].astype(np.float64)

    # save processed data to file
    data.to_csv(processed_data_path, index=False)

    # calculate metadata
    num_users = len(data[user_column_name].unique())
    num_items = len(data[item_column_name].unique())
    num_interactions = len(data)
    feedback_type = "explicit" if rating_column_name in data.columns else "implicit"
    user_counter = Counter(data[user_column_name])
    item_counter = Counter(data[item_column_name])
    meta_data = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "density": num_interactions / (num_users * num_items) * 100,
        "feedback_type": feedback_type,
        "user_item_ratio": num_users / num_items,
        "item_user_ratio": num_items / num_users,
        "highest_num_rating_by_single_user": user_counter.most_common()[0][1],
        "lowest_num_rating_by_single_user": user_counter.most_common()[-1][1],
        "highest_num_rating_on_single_item": item_counter.most_common()[0][1],
        "lowest_num_rating_on_single_item": item_counter.most_common()[-1][1],
        "mean_num_ratings_by_user": num_interactions / num_users,
        "mean_num_ratings_on_item": num_interactions / num_items
    }

    # save metadata
    with processed_data_metadata_path.open('w') as file:
        json.dump(meta_data, file, indent=4)

    # generate path for atomic conversion
    Path.mkdir(Path(atomic_folder_path), exist_ok=True)

    # rename columns
    rename_dict = {user_column_name: "user_id:token", item_column_name: "item_id:token"}
    if rating_column_name in list(data):
        rename_dict[rating_column_name] = "rating:float"
    if timestamp_column_name in list(data):
        data = data.drop(columns=[timestamp_column_name]).rename(columns=rename_dict)
    else:
        data = data.rename(columns=rename_dict)
    user_column_name = "user_id:token"
    item_column_name = "item_id:token"
    rating_column_name = "rating:float"
    timestamp_column_name = None

    # convert data to implicit
    if feedback_type == "explicit" and not data_set_name in ["MovieLens-100K-Ratings", "MovieLens-1M-Ratings",
                                                             "Netflix"]:
        max_rating = data[rating_column_name].max()
        min_rating = data[rating_column_name].min()

        scaled_max_rating = abs(max_rating) + abs(min_rating)
        rating_cutoff = round(scaled_max_rating * 0.0) - abs(min_rating)
        data = data[data[rating_column_name] >= rating_cutoff][[user_column_name, item_column_name]]

        feedback_type = "implicit"

    # subsample large data sets
    if len(data) > 10_000_000:
        data = data.sample(n=10_000_000, random_state=42)

    # 5-core pruning
    u_cnt, i_cnt = Counter(data[user_column_name]), Counter(data[item_column_name])
    while len(data) > 0 and min(u_cnt.values()) < 5 or min(i_cnt.values()) < 5:
        u_sig = [k for k in u_cnt if (u_cnt[k] >= 5)]
        i_sig = [k for k in i_cnt if (i_cnt[k] >= 5)]
        data = data[data[user_column_name].isin(u_sig)]
        data = data[data[item_column_name].isin(i_sig)]
        u_cnt, i_cnt = Counter(data[user_column_name]), Counter(data[item_column_name])
    u_sig = [k for k in u_cnt if (u_cnt[k] < len(data[item_column_name].unique()))]
    data = data[data[user_column_name].isin(u_sig)]

    # normalize identifiers
    for col in [user_column_name, item_column_name]:
        unique_ids = {key: value for value, key in enumerate(data[col].unique())}
        data[col].update(data[col].map(unique_ids))

    # save atomic data
    data.to_csv(atomic_data_path, index=False)

    # calculate data splits
    data_splits = {}
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(folds.split(data)):
        train, test = data.iloc[train_index], data.iloc[test_index]
        train, valid = train_test_split(train, test_size=0.2 / (1 - (1 / 5)), random_state=42)
        data_splits[f"fold_{fold}"] = {"train": train, "valid": valid, "test": test}

    # save data splits
    data_splits[0]["train"].to_csv(f"{atomic_folder_path}/{data_set_name}.train_split_{0}.inter", index=False)
    data_splits[0]["valid"].to_csv(f"{atomic_folder_path}/{data_set_name}.valid_split_{0}.inter", index=False)
    data_splits[0]["test"].to_csv(f"{atomic_folder_path}/{data_set_name}.test_split_{0}.inter", index=False)

    # calculate metadata
    meta_data = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "density": num_interactions / (num_users * num_items) * 100,
        "feedback_type": feedback_type,
        "user_item_ratio": num_users / num_items,
        "item_user_ratio": num_items / num_users,
        "highest_num_rating_by_single_user": user_counter.most_common()[0][1],
        "lowest_num_rating_by_single_user": user_counter.most_common()[-1][1],
        "highest_num_rating_on_single_item": item_counter.most_common()[0][1],
        "lowest_num_rating_on_single_item": item_counter.most_common()[-1][1],
        "mean_num_ratings_by_user": num_interactions / num_users,
        "mean_num_ratings_on_item": num_interactions / num_items
    }

    # save metadata
    with atomic_data_metadata_path.open('w') as file:
        json.dump(meta_data, file, indent=4)

    print(f"Finished processing {data_set_name}.")
