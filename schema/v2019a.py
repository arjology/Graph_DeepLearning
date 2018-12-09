from enum import Enum
from typing import Iterable

from utils import optmap
from graph import DseGraphProperty, DsePropertyType
from schema import Person, Company, Review, GraphElement, GraphSchemaId

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

class Person_v2019a(Person):
    """Properties of a person stored in DSE graph."""

    SCHEMA = GraphSchemaId.v2019a
    FIELDS = PERSON_FIELDS

    __slots__ = tuple(
        # order must match expected Kafka message column order
        [props.name for props in PERSON_FIELDS]
    )

    @classmethod
    def label(cls) -> str:
        """Vertex label."""
        return "person"

    @classmethod
    def properties(cls) -> Iterable[str]:
        """Ordered list of property names."""
        return cls.__slots__

    @classmethod
    def from_vertex(cls, vertex: Person) -> GraphElement:
        """Initialize DSE vertex properties from Person instance."""
        cls._check_vertex(vertex)
        return cls(
            name=vertex.name,
            gender=vertex.gender,
            age=vertex.age,
            preferences=vertex.preferences,
        )

    def to_vertex(self) -> Person:
        """Create Person given DSE vertex properties."""
        # pylint: disable=E1101
        return Person(
            name=self.name,
            gender=self.gender,
            age=self.age,
            preferences=self.preferences,
        )


class Company_v2019a(Company):
    """Properties of a company stored in DSE graph."""
    SCHEMA = GraphSchemaId.v2019a
    FIELDS = COMPANY_FIELDS

    __slots__ = tuple(
        # order must match expected Kafka message column order
        [props.name for props in COMPANY_FIELDS]
    )

    @classmethod
    def label(cls) -> str:
        """Vertex label."""
        return "company"

    @classmethod
    def properties(cls) -> Iterable[str]:
        """Ordered list of property names."""
        return cls.__slots__

    @classmethod
    def from_vertex(cls, vertex: Company) -> GraphElement:
        """Initialize DSE vertex properties from Person instance."""
        cls._check_vertex(vertex)
        return cls(
            name=vertex.name,
            styles=vertex.styles,
        )

    def to_vertex(self) -> Company:
        """Create Person given DSE vertex properties."""
        # pylint: disable=E1101
        return Person(
            name=self.name,
            styles=self.styles,
        )


class Review_2019a(Review):
    """Properties of an edge stored in DSE graph."""

    SCHEMA = GraphSchemaId.v2019a
    FIELDS = REVIEW_FIELDS

    __slots__ = tuple(
        [props.name for props in REVIEW_FIELDS]
    )

    @classmethod
    def label(cls) -> str:
        """Edge label."""
        return "review"

    @classmethod
    def properties(cls) -> Iterable[str]:
        """Get ordered list of field names, order must match expectation of ETL pipeline consumer."""
        return [name for name in cls.__slots__ if name not in ("name", "company")]

    def defined(self) -> Iterable[str]:
        """Get list of defined (i.e., non-None) attributes."""
        return [name for name in self.__slots__ if name not in ("name", "company") and getattr(self, name) is not None]

    @classmethod
    def from_edge(cls, edge: Review) -> GraphElement:
        """Initialize DSE edge properties from Review edge."""
        cls._check_edge(edge)
        return cls(
            name=optmap(Person_v2019a.from_vertex, edge.name),
            company=optmap(Person_v2019a.from_vertex, edge.company),
            score=edge.score,
        )

    @classmethod
    def from_result(cls, result) -> GraphElement:
        """Initialize builder object from result of DSE graph query."""
        if hasattr(result, 'value'):
            obj = cls.from_properties(result.value["edge"]["properties"])
            obj.name = Person_v2019a.from_properties(result.value["src"]["properties"])
            obj.company = Person_v2019a.from_properties(result.value["dst"]["properties"])
        else:
            obj = cls.from_properties(result.properties)
            obj.src = Person_v2019a.from_properties(result.outV)
            obj.dst = Person_v2019a.from_properties(result.inV)
        return obj

    def to_edge(self) -> Review:
        """Create Reviews edge from DSE edge properties."""
        return Review(
            name=optmap(lambda obj: obj.to_vertex(), self.name),
            company=optmap(lambda obj: obj.to_vertex(), self.company),
            score=optmap(float, self.score)
        )


class GraphSchema_v2019a(GraphSchema):
    """Graph schema 'v2019a'."""

    @classmethod
    def uid(cls) -> GraphSchemaId:
        return GraphSchemaId.v2019a

    @classmethod
    def person(cls) -> Person:
        return Person_v2019a

    @classmethod
    def company(cls) -> Company:
        return Company_v2019a

    @classmethod
    def review(cls) -> Review:
        return Review_2019a

    @classmethod
    def describe(cls, exist_ok: bool=False) -> str:
        """Create graph schema as a single text block, including search indices."""
        schema = ""

        # labels
        person_label = cls.person().label()
        company_label = cls.company().label()
        review_label = cls.review().label()

        # Person properties
        person_props = [
            DseGraphProperty(name=PERSON_FIELDS.name.name,
                             type=DsePropertyType.Text),
            DseGraphProperty(name=PERSON_FIELDS.gender.name,
                             type=DsePropertyType.Text),
            DseGraphProperty(name=PERSON_FIELDS.age.name,
                             type=DsePropertyType.Int),
            DseGraphProperty(name=PERSON_FIELDS.preferences.name,
                             type=DsePropertyType.List),
        ]

        # Company properties
        company_props = [
            DseGraphProperty(name=COMPANY_FIELDS.name.name,
                             type=DsePropertyType.Text),
            DseGraphProperty(name=COMPANY_FIELDS.styles.name,
                             type=DsePropertyType.List),
        ]

        # GeoImageMatches properties
        review_props = [
            DseGraphProperty(name=REVIEW_FIELDS.name.name,
                             type=DsePropertyType.Text),
            DseGraphProperty(name=REVIEW_FIELDS.company.name,
                             type=DsePropertyType.Text),
            DseGraphProperty(name=REVIEW_FIELDS.score.name,
                             type=DsePropertyType.Double),
        ]

        # define vertices
        vertices = [
            "schema.vertexLabel('" + person_label + "')"
            + ".properties(" + ", ".join([repr(prop.name) for prop in person_props]) + ")"
            + (".ifNotExists()" if exist_ok else "")
            + ".create()",
            "schema.vertexLabel('" + company_label + "')"
            + ".properties(" + ", ".join([repr(prop.name) for prop in company_props]) + ")"
            + (".ifNotExists()" if exist_ok else "")
            + ".create()"
            ]

        # define edges
        edges = [
            "schema.edgeLabel('" + review_label + "').single()"
            + ".properties(" + ", ".join([repr(prop.name) for prop in review_props]) + ")"
            + ".connection('{0}', '{1}')".format(person_label, company_label)
            + (".ifNotExists()" if exist_ok else "")
            + ".create()"
            ]

        # define indices
        indices = [
            "schema.vertexLabel('" + person_label + "')"
            + ".index('byName').materialized().by('name')"
            + (".ifNotExists()" if exist_ok else "")
            + ".add()",
            "schema.vertexLabel('" + company_label + "')"
            + ".index('byName').materialized().by('name')"
            + (".ifNotExists()" if exist_ok else "")
            + ".add()",
            "schema.vertexLabel('" + person_label + "')"
            + ".index('toCompanyRated').outE('" + review_label + "').by('score')"
            + ".add()",
            "schema.vertexLabel('" + company_label + "')"
            + ".index('toPersonWhoRated').inE('" + review_label + "').by('score')"
            + ".add()"            
        ]

        props = set(
            [prop.describe(exist_ok=exist_ok) for prop in person_props]
            + [prop.describe(exist_ok=exist_ok) for prop in company_props]
            + [prop.describe(exist_ok=exist_ok) for prop in review_props]
            )

        schema += "\n".join(
            list(props)
            + vertices
            + edges
            + indices
        )

        return schema
