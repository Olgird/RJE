import urllib
from typing import List

from SPARQLWrapper import JSON, SPARQLWrapper


class KnowledgeGraphFreebase:
    def __init__(self) -> None:
        self.sparql = SPARQLWrapper("http://localhost:3001/sparql")
        self.sparql.setReturnFormat(JSON)

    # usage: load_raw_dataset.py get_answer_names.py
    def get_ent_name(self, id: str) -> str:
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?name WHERE {{
                    :{id} :type.object.name ?name .
                    FILTER(LANGMATCHES(LANG(?name), 'en'))
                }}
        """
        # FILTER(lang(?name) = "en")

        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        if not results["results"]["bindings"]:
            return None

        return results["results"]["bindings"][0]["name"]["value"]

    # usage: get_retriever_train_data.py run_get_localgraph.py
    def get_relation(self, src: str, limit: int = 200) -> List[str]:
        src = ":" + src
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 WHERE {{
                    {src} ?r0_ ?t0 .
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?t0, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(:)) as ?r0)
                }} LIMIT {limit}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [
            i["r0"]["value"]
            for i in results["results"]["bindings"]
            if i["r0"]["value"] != "type.object.type"
        ]

    # usage: run_get_localgraph.py
    def get_tail(self, src: str, relation, limit: int) -> List[str]:
        src = ":" + src
        relation = ":" + relation
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?t0 WHERE {{
                    {src} {relation} ?t0_ .
                    FILTER regex(?t0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?t0_),str(:)) as ?t0)
                }}LIMIT {limit}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [i["t0"]["value"] for i in results["results"]["bindings"]]

