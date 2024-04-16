import json
import pandas as pd

from pathlib import Path

data_sets = ['Amazon2018-Books', 'Amazon2018-CDs-And-Vinyl', 'Amazon2018-Clothing-Shoes-And-Jewelry',
             'Amazon2018-Electronics', 'Amazon2018-Sports-And-Outdoors', 'Amazon2018-Toys-And-Games', 'Gowalla',
             'Hetrec-LastFM', 'MovieLens-100K', 'MovieLens-1M', 'MovieLens-Latest-Small', 'Netflix', 'Retailrocket',
             'Yelp-2018']

data = {}
for data_set_name in data_sets:
    metadata_file = Path(f"./data_sets/{data_set_name}/atomic/metadata.json")
    with open(metadata_file, "r") as file:
        metadata = json.load(file)
        data[data_set_name] = metadata
data = pd.DataFrame(data).T
data.drop(columns=["user_item_ratio", "item_user_ratio", "highest_num_rating_by_single_user",
                   "lowest_num_rating_by_single_user", "highest_num_rating_on_single_item",
                   "lowest_num_rating_on_single_item", "feedback_type"], inplace=True)
data.rename(columns={"num_users": "Users", "num_items": "Items", "num_interactions": "Interactions",
                     "density": "Density", "mean_num_ratings_by_user": "Mean Ratings by User",
                     "mean_num_ratings_on_item": "Mean Ratings on Item"}, inplace=True)
data["Users"] = data["Users"].apply(lambda x: f'{x:,}')
data["Items"] = data["Items"].apply(lambda x: f'{x:,}')
data["Interactions"] = data["Interactions"].apply(lambda x: f'{x:,}')
latex = data.to_latex()
data.drop(columns=["Mean Ratings by User", "Mean Ratings on Item"], inplace=True)
latex_small = data.to_latex()
