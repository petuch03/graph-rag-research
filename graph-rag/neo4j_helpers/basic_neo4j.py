from neo4j import GraphDatabase


class BasicNeo4j:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def erase_database(self):
        with self.driver.session() as session:
            session.execute_write(self._erase_all)

    @staticmethod
    def _erase_all(tx):
        query = "MATCH (n) DETACH DELETE n"
        tx.run(query)

    def execute_custom(self, cypher_query):
        query_type = "insert" if "CREATE" in cypher_query or "MERGE" in cypher_query else "extract"

        with self.driver.session() as session:
            result = session.run(cypher_query)

            if query_type == "insert":
                return "Data insertion successful."
            else:
                return [record.data() for record in result]

