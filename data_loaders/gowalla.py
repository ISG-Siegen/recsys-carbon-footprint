import pandas as pd

from .loader import Loader


class Gowalla(Loader):

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        return pd.read_csv(f"{source_path}/loc-gowalla_totalCheckins.txt.gz", compression="gzip",
                           names=[user_column_name, timestamp_column_name, "latitude", "longitude", item_column_name],
                           usecols=[user_column_name, item_column_name, timestamp_column_name], header=None, sep="\t")
