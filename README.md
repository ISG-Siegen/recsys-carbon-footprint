# RecSys Carbon Footprint

This repository contains the code used to produce the results in our paper "From Clicks to Carbon: The Environmental Toll of Recommender Systems".

## Usage

1. Install the required packages with `pip install -r requirements.txt`.
2. Download the data sets by using the data and place it in the `source/files` directory.

|                                       | 0                                                                    |
|:--------------------------------------|:---------------------------------------------------------------------|
| MovieLens-100K                        | https://grouplens.org/datasets/movielens/                            |
| MovieLens-1M                          | https://grouplens.org/datasets/movielens/                            |
| MovieLens-Latest-Small                | https://grouplens.org/datasets/movielens/                            |
| Hetrec-LastFM                         | https://grouplens.org/datasets/hetrec-2011/                          |
| Gowalla                               | https://snap.stanford.edu/data/loc-Gowalla.html                      |
| Amazon2018-Toys-And-Games             | https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html     |
| Amazon2018-Electronics                | https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html     |
| Amazon2018-Sports-And-Outdoors        | https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html     |
| Amazon2018-Clothing-Shoes-And-Jewelry | https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html     |
| Amazon2018-CDs-And-Vinyl              | https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html     |
| Amazon2018-Books                      | https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html     |
| Yelp-2018                             | https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/1 |
| Retailrocket                          | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset       |
| MovieLens-100K-Ratings                | https://grouplens.org/datasets/movielens/                            |
| MovieLens-1M-Ratings                  | https://grouplens.org/datasets/movielens/                            |

3. (Optional) Check instructions in the `power_logger` directory to run the hardware energy meter logging.
4. Run `preprocessing.py` to preprocess the data sets.
5. Change the loaded `experiments_xx.json` in `energy_tester.py` to run the experiment of your choice.
6. Run `energy_tester.py` to run the experiments.
