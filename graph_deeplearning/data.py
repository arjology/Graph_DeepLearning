import logging
import random
import pandas as pd
import numpy as np
from faker import Faker

from graph_deeplearning.graph import GraphMode, Person, Company, Review
from graph_deeplearning.graph.dse import DseGraph

class DataMaker:

    def __init__(self, N_people, N_companies, N_reviews_per_person, N_styles, logger):

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
        self.ages = np.random.randint(low=18, high=70, size=self.N_people)
        self.genders = ['m', 'f']
        self.styles = range(N_styles)

    def generate_companies(self):
        """Generate a unique set of companies"""
        self.logger.debug("*** Generating {} companies...".format(self.N_companies))
        company_idx = 0
        while len(self.company_names) < self.N_companies:
            name = self.fake.company()
            styles = np.random.binomial(1, 0.5, 6)
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
        
        self.logger.debug("\tpeople: {}".format(len(self.people)))
        self.logger.debug("\treviews: {}".format(len(self.reviews)))

    def build_dataframes(self) -> List[pd.DataFrame]:
        self.logger.debug("Building DataFrames...")

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
        self.logger.debug("Loading data sets...")
        graph = DseGraph("companyReviews", GraphMode.READ_WRITE)

        people = [graph.schema.person().from_dict(person._asdict()) for person in self.people]
        companies = [graph.schema.company().from_dict(company._asdict()) for company in self.companies]
        reviews = [graph.schema.review().from_dict(review._asdict()) for review in self.reviews]
        
        graph.add_person(*people)
        graph.add_company(*companies)
        graph.add_review(*reviews)
                