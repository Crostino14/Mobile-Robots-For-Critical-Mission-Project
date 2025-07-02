import yaml
import math

class Graph:
    """
    Represents a graph with nodes and connections, initialized from YAML files.

    Attributes:
        graph (dict): The graph structure loaded from a YAML file.
        points (dict): The points coordinates loaded from a YAML file.

    Methods:
        __init__(self, graph_file_path, points_file_path):
            Initializes the Graph object by loading the graph structure and points coordinates from YAML files.

        load_yaml(self, file_path):
            Loads data from a YAML file.

        find_nearest_point(self, input_point):
            Find the nearest point in the graph to the given input point.

        get_next_node_and_orientation(self, current_orientation, direction, current_node):
            Retrieves the next node and orientation based on the current orientation, direction, and node.

        get_coordinates(self, node):
            Retrieves the coordinates of a given node.

        orientation_landmark_conversion(self, orientation):
            Converts orientation strings to landmark strings used in graph navigation.
    """
    def __init__(self, graph_file_path, points_file_path):
        """
        Initializes the Graph object by loading the graph structure and points coordinates from YAML files.

        Args:
            graph_file_path (str): The file path to the YAML file containing the graph structure.
            points_file_path (str): The file path to the YAML file containing the points coordinates.
        """
        self.graph = self.load_yaml(graph_file_path)
        self.points = self.load_yaml(points_file_path)

    def load_yaml(self, file_path):
        """
        Loads data from a YAML file.

        Args:
            file_path (str): The file path to the YAML file.

        Returns:
            dict: The data loaded from the YAML file.
        """
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def find_nearest_point(self, input_point):
        """
        Find the nearest point in the graph to the given input point.

        Args:
            input_point (dict): Input point with 'x' and 'y' keys representing coordinates.

        Returns:
            str: Key of the nearest point in the graph.
        """
        min_distance = float('inf')
        nearest_point = None

        for main_key, sub_points in self.points.items():
            for sub_key, value in sub_points.items():
                distance = math.sqrt((input_point['x'] - value['x']) ** 2 + (input_point['y'] - value['y']) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = main_key

        return nearest_point       

    def get_next_node_and_orientation(self, current_orientation, direction, current_node):
        """
        Retrieves the next node and orientation based on the current orientation, direction, and node.

        Args:
            current_orientation (str): The current orientation (NORTH, EAST, ecc...).
            direction (str): The direction to move in the graph.
            current_node (str): The current node identifier in the graph.

        Returns:
            tuple: A tuple containing the coordinates of the next node (dict) and the next orientation (str).
                   If the path is not found, returns (None, None).
        """
        try:
            next_node = self.graph[current_orientation][direction][current_node]
            next_orientation = self.graph[current_orientation][direction]['NextOrientation']
            landmark = self.orientation_landmark_conversion(next_orientation)
            return self.points[next_node][landmark], next_orientation
        except KeyError as e:
            return None, None

    def get_coordinates(self, node):
        """
        Retrieves the coordinates of a given node.

        Args:
            node (str): The node identifier.

        Returns:
            tuple: A tuple containing the x and y coordinates of the node.
                   If the node is not found, returns (None, None).
        """
        try:
            return node['x'], node['y']
        except KeyError:
            return None, None
    
    def orientation_landmark_conversion(self, orientation):
        """
        Converts orientation strings to landmark strings used in graph navigation.

        Args:
            orientation (str): Orientation string (NORTH, EAST, SOUTH, WEST).

        Returns:
            str: Corresponding landmark string ('UP' for NORTH, 'SX' for EAST, 'DOWN' for SOUTH, 'DX' for WEST).
        """        
        if orientation == "NORTH":
            return "DOWN"
        elif orientation == "EAST":
            return "SX"
        elif orientation == "SOUTH":
            return "UP"
        elif orientation == "WEST":
            return "DX"
