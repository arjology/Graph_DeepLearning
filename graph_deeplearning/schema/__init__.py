from typing import NamedTuple
from enum import Enum

field_to_DSE = {
    'list': 'list',
    'str': 'text',
    'float': 'float',
    'int': 'int'
}

class PERSON_FIELDS(Enum):
    """Enumeration of Person fields"""
    name = 0
    gender = 1
    age = 2
    preferences = 3


class COMPANY_FIELDS(Enum):
    """Enumeration of Company fields"""
    name = 0
    style = 1


class REVIEW_FIELDS(Enum):
    """Enumeration of Review fields"""
    name = 0
    company = 1
    score = 2

Person = NamedTuple('Person', [
    ('name', str),
    ('gender', str),
    ('age', int),
    ('preferences', list)
])
Person.__new__.__defaults__ = tuple([None]*len(PERSON_FIELDS))

Company = NamedTuple('Company', [
    ('name', str),
    ('styles', list)
])
Company.__new__.__defaults__ = tuple([None]*len(COMPANY_FIELDS))

Review = NamedTuple('Review', [
    ('person', Person),
    ('company', Company),
    ('score', float)
])
Review.__new__.__defaults__ = tuple([None]*len(REVIEW_FIELDS))

def add_person_query(g, person: NamedTuple):
    query = "graph.addVertex(label, 'person',"
    query += ','.join(("'{k}',{k}".format(k=k)) for k,_ in person._asdict().items())
    query += ")"
    params = {}
    params.update(dict([(str(k),','.join(map(str, v)) if isinstance(v, list) else v) for k,v in person._asdict().items()]))
    # print("\n{}\n{}".format(query, params))
    g.execute_graph(query, params)

def add_company_query(g, company: NamedTuple):
    query = "graph.addVertex(label, 'company',"
    query += ','.join(("'{k}',{k}".format(k=k)) for k,_ in company._asdict().items())
    query += ")"
    params = {}
    params.update(dict([(str(k),','.join(map(str, v)) if isinstance(v, list) else v) for k,v in company._asdict().items()]))
    g.execute_graph(query, params)

def add_review_query(g, review: NamedTuple):
    src = "graph.addVertex(label, 'person',"
    src += ','.join(("'{k}',src_{k}".format(k=k)) for k,_ in review.person._asdict().items())
    src += ")"

    dst = "graph.addVertex(label, 'company',"
    dst += ','.join(("'{k}',dst_{k}".format(k=k)) for k,v in review.company._asdict().items())
    dst += ")"

    query = """
    src = {src}
    dst = {dst}
    src.addEdge('review', dst, 'score', score)
    """.format(
        src=src,
        dst=dst,

    )
    params = {'score': review.score}
    params.update(dict([("src_"+str(k),','.join(map(str, v)) if isinstance(v, list) else v) for k,v in review.person._asdict().items()]))
    params.update(dict([("dst_"+str(k),','.join(map(str, v)) if isinstance(v, list) else v) for k,v in review.company._asdict().items()]))

    g.execute_graph(query, params)

def create_graph(g, graph_name: str=None):
    from dse.cluster import EXEC_PROFILE_GRAPH_SYSTEM_DEFAULT

    g.execute_graph(
        "system.graph(name).ifNotExists().create();",
        {'name': graph_name},
        execution_profile=EXEC_PROFILE_GRAPH_SYSTEM_DEFAULT
    )

def create_schema(g, graph_name: str=None):    

    properties = """
        schema.propertyKey('{name}').{name_value_type}().{name_multiplicity}().ifNotExists().create();
        schema.propertyKey('{gender}').{gender_value_type}().{gender_multiplicity}().ifNotExists().create();
        schema.propertyKey('{age}').{age_value_type}().{age_multiplicity}().ifNotExists().create();
        schema.propertyKey('{preferences}').{preferences_value_type}().{preferences_multiplicity}().ifNotExists().create();
        schema.propertyKey('{styles}').{styles_value_type}().{styles_multiplicity}().ifNotExists().create();
        schema.propertyKey('{score}').{score_value_type}().{score_multiplicity}().ifNotExists().create();
    """.format(
        name="name",
        name_value_type="Text",
        name_multiplicity="single",
        gender="gender",
        gender_value_type="Text",
        gender_multiplicity="single",
        age="age",
        age_value_type="Int",
        age_multiplicity="single",
        preferences="preferences",
        preferences_value_type="Text",
        preferences_multiplicity="single",
        styles="styles",
        styles_value_type="Text",
        styles_multiplicity="single",
        score="score",
        score_value_type="Float",
        score_multiplicity="single",
    )
    g.execute_graph(properties)

    vertices = """
        schema.vertexLabel('{person_lbl}').properties('{name}', '{gender}', '{age}', '{preferences}').ifNotExists().create();
        schema.vertexLabel('{company_lbl}').properties('{name}', '{styles}').ifNotExists().create();
    """.format(
        name="name",
        gender="gender",
        age="age",
        preferences="preferences",
        styles="styles",
        person_lbl="person",
        company_lbl="company",        
    )
    g.execute_graph(vertices)

    edges = """
        schema.edgeLabel('{review_lbl}').single().properties('{score}').connection('{person_lbl}', '{company_lbl}').ifNotExists().create();
    """.format(
        score="score",
        review_lbl="review",
        person_lbl="person",
        company_lbl="company",        
    ) 
    g.execute_graph(edges)
    