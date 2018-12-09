from faker import Faker
import numpy as np
from typing import NamedTuple, List
import random
import argparse
import pandas as pd
import logging

from utils import optmap, select, Person, Company, Review
from graph import DseConfig, DseGraph, GraphMode

LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
LOG_LEVEL = logging.DEBUG
logger = logging.getLogger("DataMaker")
handler = logging.StreamHandler()
formatter = logging.Formatter(LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

class DataMaker:

    def __init__(self, N_people, N_companies, N_reviews_per_person, N_styles):

        self.fake = Faker()

        # Number of people, products, and reviews per person
        self.N_people = N_people
        self.N_companies = N_companies
        self.N_reviews_per_person = N_reviews_per_person
        self.N_styles = N_styles
        self.N_reviews = self.N_people*self.N_reviews_per_person

        # Instantiate collections of companies, people, and reviews
        self.people = [None]*self.N_people
        self.reviews = [None]*self.N_reviews
        self.companies = [None]*self.N_companies
        self.company_names = set()

        # Instantiate attributes ages, genders, and features
        self.ages = np.random.randint(low=18, high=70, size=self.N_people)
        self.genders = ['m', 'f']
        self.styles = range(N_styles)

    def generate_companies(self):
        """Generate a unique set of companies"""
        logger.debug("*** Generating {} companies...".format(self.N_companies))
        company_idx = 0
        while len(self.company_names) < self.N_companies:
            name = self.fake.company()
            styles = np.random.binomial(1, 0.5, 6)
            company = Company(name, styles)
            if company.name not in self.company_names:
                self.companies[company_idx] = company
                self.company_names.add(company.name)
                company_idx += 1
        logger.debug("\tcompanies: {}".format(len(self.companies)))

    def generate_people_and_reviews(self):
        """Instantite product categories"""
        logger.debug("*** Generating {} people and {} reviews per person...".format(self.N_people, 
                                                                                self.N_reviews_per_person))
        for person_idx in range(self.N_people):
                p_gender = round(np.random.rand())
                person = Person(self.fake.name(),
                                self.ages[person_idx],
                                self.genders[p_gender],
                                np.random.binomial(1, 0.5, 6)
                        )
                self.people[person_idx] = person

                review_idx = self.N_reviews_per_person*person_idx
                for company_idx in range(self.N_reviews_per_person):
                    company = random.choice(tuple(self.companies))
                    score = float(np.dot(person.preferences, company.styles)/self.N_styles)
                    review = Review(person.name, company.name, score)
                    self.reviews[review_idx+company_idx] = review
        
        logger.debug("\tpeople: {}".format(len(self.people)))
        logger.debug("\treviews: {}".format(len(self.reviews)))

    def build_dataframes(self) -> List[pd.DataFrame]:
        logger.debug("Building DataFrames...")

        styles_cols = ['P{}'.format(i) for i in range(self.N_styles)]
        people_df = pd.DataFrame(columns=['Name','Age','Gender']+styles_cols,
                                index=list(range(self.N_people))
        )
        companies_df = pd.DataFrame(columns=['Name']+styles_cols,
                                    index=list(range(self.N_companies))
        )
        reviews_df = pd.DataFrame(columns=['Name','Company','Score'],
                                index=list(range(self.N_reviews))
        )

        for person_idx in range(self.N_people):
            person = self.people[person_idx]
            p_props = {'Name':person.name, 'Gender':person.gender, 'Age':person.age}
            p_props.update(dict(('P{}'.format(i), p) for i, p in enumerate(person.preferences)))
            people_df.loc[person_idx] = pd.Series(p_props)

        for company_idx in range(self.N_companies):
            company = self.companies[company_idx]
            c_props = {'Name':company.name}
            c_props.update(dict(('P{}'.format(p), p) for p in company.styles))
            companies_df.loc[company_idx] = pd.Series(c_props)

        for review_idx in range(self.N_reviews):
            review = self.reviews[review_idx]
            r_props = {'Name':review.name, 'Company':review.company, 'Score':review.score}
            reviews_df.loc[review_idx] = pd.Series(r_props)

        return people_df, companies_df, reviews_df

    def load(self):
        logger.debug("Loading data sets...")
        graph = DseGraph("companyReviews", GraphMode.READ_WRITE)

        people = [self.graph.schema.person().from_dict(person._asdict()) for person in self.people]
        companies = [self.graph.schema.company().from_dict(company._asdict()) for company in self.companies]
        reviews = [self.graph.schema.review().from_dict(review._asdict()) for review in self.reviews]
        
        graph.add_person(*people)
        graph.add_company(*companies)
        graph.add_review(*reviews)       
            

def main(argv=None):

    parser = argparse.ArgumentParser(argv)
    parser.add_argument('--people', dest="N_people", type=int)
    parser.add_argument('--companies', dest="N_companies", type=int)
    parser.add_argument('--reviews', dest="N_reviews_per_person", type=int)
    parser.add_argument('--styles', dest="N_styles", type=int)
    parser.add_argument('--plot', dest="plot", type=bool)
    parser.add_argument('--save', dest="save", type=bool)
    parser.add_argument('--load', dest="load", type=bool)
    args = parser.parse_args()

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

    data_maker = DataMaker(N_people, N_companies, N_reviews_per_person, N_styles)
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