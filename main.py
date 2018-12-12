from faker import Faker
import numpy as np
from typing import NamedTuple, List
import random
import argparse
import pandas as pd
import logging



from graph_deeplearning.data import DataMaker
from graph_deeplearning.utilities import optmap, select, PathType
from graph_deeplearning.schema import Person, Company, Review


def main(argv=None):

    LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    LOG_LEVEL = logging.DEBUG
    logger = logging.getLogger("graph_deeplearning")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)

    parser = argparse.ArgumentParser(argv)
    parser.add_argument('--people', dest="N_people", type=int)
    parser.add_argument('--companies', dest="N_companies", type=int)
    parser.add_argument('--reviews', dest="N_reviews_per_person", type=int)
    parser.add_argument('--styles', dest="N_styles", type=int)
    parser.add_argument('--plot', dest="plot", type=bool)
    parser.add_argument('--save', dest="save", type=bool)
    parser.add_argument('--load', dest="load", type=bool)
    parser.add_argument('--conf', dest="conf", type=PathType(exists=True, type='file'))
    args = parser.parse_args()

    print("Configuration [{}]: {}".format(type(args.conf), args.conf))
    if args.conf:
        from pyhocon import ConfigFactory
        config = ConfigFactory.parse_file(conf)
    else:
        from graph_deeplearning.utilities import DEFAULT_CONFIG as config

    N_people = select(args.N_people, 250)
    N_companies = select(args.N_companies, 50)
    N_reviews_per_person = select(args.N_reviews_per_person, 40)
    N_styles = select(args.N_styles, 6)

    summary = """
    {header}
    # Number of people:\t\t\t{people}
    # Number of reviews per person:\t{rpp}
    # Number of companies:\t\t{companies}
    # Number of reviews:\t\t{reviews}
    # Number of styles:\t\t\t{styles}
    {footer}
    """.format(people=N_people, 
               rpp=N_reviews_per_person,
               companies=N_companies,
               reviews=N_people*N_reviews_per_person,
               styles=N_styles,
               header=''.join(['#']*40),
               footer=''.join(['#']*40)
    )
    
    print(summary)

    plot = select(args.plot, False)
    save = select(args.save, False)
    load = select(args.load, False)

    data_maker = DataMaker(
        N_people=N_people,
        N_companies=N_companies,
        N_reviews_per_person=N_reviews_per_person,
        N_styles=N_styles,
        logger=logger.getChild("DataMaker"),
        config=config
    )
    data_maker.generate_companies()
    data_maker.generate_people_and_reviews()

    if plot:
        logger.debug("Plotting data sets...")
        import matplotlib.pyplot as plt
        import seaborn as sns

    if save:
        logger.debug("Saving data sets...")
        people_df, companies_df, reviews_df = data_maker.build_dataframes()
        people_df.to_csv('data/people.csv', header=False, index=False)
        companies_df.to_csv('data/companies.csv', header=False, index=False)
        reviews_df.to_csv('data/reviews.csv', header=False, index=False)

    if load:
        data_maker.load()

if __name__ == "__main__":
    main()