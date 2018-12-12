import logging
import random
import pandas as pd
import numpy as np
from faker import Faker
from progressbar import ProgressBar
from pyhocon import ConfigTree
from logging import Logger

from typing import List

from graph_deeplearning.schema import Person, Company, Review
from graph_deeplearning.schema import add_company_query, add_person_query, add_review_query,\
    create_graph, create_schema
from graph_deeplearning.utilities.connection import dse_get_session

class DataMaker:

    def __init__(self,
        N_people: int=None, 
        N_companies: int=None,
        N_reviews_per_person: int=None, 
        N_styles: int=None, 
        config: ConfigTree=None,
        logger: Logger=None
        ):

        if config is None:
            from graph_deeplearning.utilities import DEFAULT_CONFIG as config
        self.config = config

        if logger is None:
                import logging
                LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
                LOG_LEVEL = logging.DEBUG
                self.logger = logging.getLogger("DataMaker")
                handler = logging.StreamHandler()
                formatter = logging.Formatter(LOG_FORMAT)
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(config['default']['log_level'])
        else:
            self.logger = logger

        self.fake = Faker()
        self.logger = logger

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
        self.ages = np.random.randint(low=18, high=70, size=self.N_people).tolist()
        self.genders = ['m', 'f']
        self.styles = range(N_styles)

    def generate_companies(self):
        """Generate a unique set of companies"""
        self.logger.debug("*** Generating {} companies...".format(self.N_companies))
        company_idx = 0
        while len(self.company_names) < self.N_companies:
            name = self.fake.company()
            styles = np.random.binomial(1, 0.5, 6).tolist()
            company = Company(name, styles)
            if company.name not in self.company_names:
                self.companies[company_idx] = company
                self.company_names.add(company.name)
                company_idx += 1
        self.logger.debug("\tcompanies: {}".format(len(self.companies)))

    def generate_people_and_reviews(self):
        """Instantite product categories"""
        self.logger.debug("*** Generating {} people and {} reviews per person...".format(self.N_people, 
                                                                                self.N_reviews_per_person))
        for person_idx in range(self.N_people):
                p_gender = round(np.random.rand())
                person = Person(self.fake.name(),
                                self.genders[p_gender],
                                self.ages[person_idx],
                                np.random.binomial(1, 0.5, 6).tolist()
                        )
                self.people[person_idx] = person

                review_idx = self.N_reviews_per_person*person_idx
                for company_idx in range(self.N_reviews_per_person):
                    company = random.choice(tuple(self.companies))
                    score = (np.dot(person.preferences, company.styles)/self.N_styles)
                    review = Review(person, company, score)
                    self.reviews[review_idx+company_idx] = review
        
        self.logger.debug("\tpeople: {}".format(len(self.people)))
        self.logger.debug("\treviews: {}".format(len(self.reviews)))

    def build_dataframes(self) -> List[pd.DataFrame]:
        self.logger.debug("Building DataFrames...")

        styles_cols = ['P{}'.format(i) for i in range(self.N_styles)]
        people_df = pd.DataFrame(columns=['name','gender','age']+styles_cols,
                                index=list(range(self.N_people))
        )
        companies_df = pd.DataFrame(columns=['name']+styles_cols,
                                    index=list(range(self.N_companies))
        )
        reviews_df = pd.DataFrame(columns=['person','company','score'],
                                index=list(range(self.N_reviews))
        )

        for person_idx in range(self.N_people):
            person = self.people[person_idx]
            p_props = {'Name':person.name, 'Gender':person.gender, 'Age':person.age}
            p_props.update(dict(('P{}'.format(i), p) for i, p in enumerate(person.preferences)))
            people_df.loc[person_idx] = pd.Series(p_props)

        for company_idx in range(self.N_companies):
            company = self.companies[company_idx]
            c_props = {'name':company.name}
            c_props.update(dict(('P{}'.format(p), p) for p in company.styles))
            companies_df.loc[company_idx] = pd.Series(c_props)

        for review_idx in range(self.N_reviews):
            review = self.reviews[review_idx]
            r_props = {'person':review.name, 'company':review.company, 'score':review.score}
            reviews_df.loc[review_idx] = pd.Series(r_props)

        return people_df, companies_df, reviews_df

    def load(self):
        self.logger.debug("Loading data sets...")
        
        username = self.config['default']['dse']['username']
        password = self.config['default']['dse']['password']
        cluster_ip = self.config['default']['dse']['cluster_ip']
        graph_name = self.config['default']['dse']['graph_name']
        session = dse_get_session(username=username, password=password, cluster_ip=cluster_ip)
        self.logger.debug("Creating graph...")
        create_graph(g=session, graph_name=graph_name)

        g = dse_get_session(username=username, password=password, cluster_ip=cluster_ip, graph_name=graph_name)
        self.logger.debug("Creating schema...")
        create_schema(g=g, graph_name=graph_name)

        self.logger.debug("Uploading people...")
        pb = ProgressBar(max_value=self.N_people)
        pb.start()
        for i, person in enumerate(self.people):
            add_person_query(g, person)
            pb.update(i+1)
        pb.finish

        self.logger.debug("Uploading companies...")
        pb = ProgressBar(max_value=self.N_companies)
        pb.start()
        for i, company in enumerate(self.companies):
            add_company_query(g, company)
            pb.update(i+1)
        pb.finish

        self.logger.debug("Uploading reviews...")
        pb = ProgressBar(max_value=self.N_reviews)
        pb.start()
        for i, review in enumerate(self.reviews):
            add_review_query(g, review)
            pb.update(i+1)
        pb.finish

