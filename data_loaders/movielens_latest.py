from .movielens_large import MovieLensLarge


class MovieLensLatest(MovieLensLarge):

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        version = "latest"
        return super(MovieLensLatest, MovieLensLatest).load_from_file(source_path, user_column_name, item_column_name,
                                                                      rating_column_name, timestamp_column_name,
                                                                      version=version)
