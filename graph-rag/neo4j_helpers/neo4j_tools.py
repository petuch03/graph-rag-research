from .basic_neo4j import BasicNeo4j


class Neo4jTools:
    def __init__(self, neo4j_app: BasicNeo4j):
        self.neo4j_app = neo4j_app

    def find_all_labels(self, query="") -> str:
        query_for_retrieving_all_nodes = "CALL db.labels() YIELD label RETURN label"
        labels_json = self.neo4j_app.execute_custom(cypher_query=query_for_retrieving_all_nodes)
        labels_list = [item['label'] for item in labels_json]
        labels = ', '.join(labels_list)
        return labels

    def find_all_relationship_types(self, query: str = "") -> str:
        query_for_retrieving_all_relationship_types = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        relationship_types_json = self.neo4j_app.execute_custom(
            cypher_query=query_for_retrieving_all_relationship_types)
        relationship_types_list = [item['relationshipType'] for item in relationship_types_json]
        relationship_types = ', '.join(relationship_types_list)
        return relationship_types

    # def find_similarity_nodes(self, query: str = "") -> str:
    #     query_for_retrieving_all_relationship_types = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
    #     relationship_types_json = self.neo4j_app.execute_custom(
    #         cypher_query=query_for_retrieving_all_relationship_types)
    #     relationship_types_list = [item['relationshipType'] for item in relationship_types_json]
    #     relationship_types = ', '.join(relationship_types_list)
    #     return relationship_types

    def execute_query(self, query: str = "") -> str:
        json_response = self.neo4j_app.execute_custom(query)
        return json_response
