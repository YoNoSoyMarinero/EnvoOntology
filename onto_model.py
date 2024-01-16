import owlready2 as owl
import os

owl.onto_path.append(r"C:\Users\1milo\Desktop\Eco-ontology\Eco-ontology\envo.owl")
onto = owl.get_ontology(r"C:\Users\1milo\Desktop\Eco-ontology\Eco-ontology\envo.owl")
onto.load()

with onto:
    class Terrene(owl.Thing):
        pass

    class Image(owl.Thing):
        pass
    
    class Empty(owl.Nothing):
        pass

    class Vegetation(Terrene):
        pass
    
    class Soil(Terrene):
        pass

    class Liquid(Terrene):
        pass

    class TreeCanopy(Vegetation):
        pass

    class Water(Liquid):
        pass

    class Rock(Soil):
        pass

    class Dirt(Soil):
        pass

    class Waterlilly(Vegetation):
        pass

    class Swamp(Liquid):
        pass

    class Grass(Vegetation):
        pass

    class Sand(Soil):
        pass

    class hasX(owl.DataProperty, owl.FunctionalProperty):
        domain = [Terrene]
        range = [int]


    class hasY(owl.DataProperty, owl.FunctionalProperty):
        domain = [Terrene]
        range = [int]


    class hasPercentage(owl.DataProperty, owl.FunctionalProperty):
        domain = [Terrene]
        range = [float]


    class hasImage(owl.ObjectProperty, owl.FunctionalProperty):
        domain = [Terrene]
        range = [Image]


    class hasImageUri(owl.DataProperty, owl.FunctionalProperty):
        domain = [Image]
        range = [str]


    class hasLongitude(owl.DataProperty, owl.FunctionalProperty):
        domain = [Image]
        range = [float]


    class hasLatitude(owl.DataProperty, owl.FunctionalProperty):
        domain = [Image]
        range = [float]



onto.save()


class SPARQL:

    prefix: str = """          
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX eco: <C:/Users/1milo/Desktop/Eco-ontology/Eco-ontology/envo.owl#>"""
    

    @classmethod
    def filter_individuals(cls, image_name ,class_name="Terrene", x1=0, x2=1e6, y1=0, y2=1e6, min_percentage=0, max_percentage=100):
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX eco: <C:/Users/1milo/Desktop/Eco-ontology/Eco-ontology/envo.owl#>

                SELECT ?instance ?x ?y ?percentage
                WHERE {{
                ?instance rdf:type/rdfs:subClassOf* eco:{class_name} .
                ?instance eco:hasX ?x .
                ?instance eco:hasY ?y .
                ?instance eco:hasPercentage ?percentage .
                FILTER (?x >= {x1} && ?x <= {x2} && ?y >= {y1} && ?y <= {y2})
                FILTER (?percentage >= {min_percentage} && ?percentage <= {max_percentage})
                FILTER(CONTAINS(STR(?instance), "{image_name}"))
                }}
            """
        return list(owl.default_world.sparql(query, error_on_undefined_entities=False))


class IndividualGenerator():
    class_map: dict = {
        0: (TreeCanopy, "tree_canopy"),
        1: (Water, "water"),
        2: (Rock, "rock"),
        3: (Dirt, "dirt"),
        4: (Sand, "sand"),
        5: (Waterlilly, "waterlilly"),
        6: (Empty, "none"),
        7: (Swamp, "swamp"),
        8: (Grass, "grass")
    }

    @classmethod
    def create_terrene_individual(cls, x: int, y: int, percentage: float, class_label: int, image_name: str) -> None:
        individual_class: object = cls.class_map[class_label]
        individual_name: str = f"{image_name}_{x}_{y}_{individual_class[1]}"
        individual: object = individual_class[0](individual_name)
        individual.hasX = x
        individual.hasY = y
        individual.hasPercentage = float(percentage)
        individual.hasImage = onto[image_name]
        onto.save()

    @classmethod
    def create_image_individual(cls, image_uri: str, longitude: float, lantitude: float) -> None:
        image: Image = Image(os.path.basename(image_uri))
        image.hasImageUri = image_uri
        image.hasLongitude = longitude
        image.hasLatitude = lantitude
        onto.save()
        