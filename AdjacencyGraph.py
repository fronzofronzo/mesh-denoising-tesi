import numpy as np
from collections import defaultdict
class AdjacencyGraph:
    """
    Optimized class to handle face adjacency.
    """
    def __init__(self, adj_pairs):
        """
        Initialize adjacency graph.

        Args:
            adj_pairs: array numpy (N,2) with couples of adjacent faces.
        """
        self.adj_pairs = adj_pairs
        self.num_faces = None
        self.adjacency_dict = {}
        self.adjacency_sets = {}
        self.neighbor_counts = None

        self._second_ring_cache = {}
        self._build_adjacency_structures()

    def _build_adjacency_structures(self):
        """
        Build data structures for adjacency in one step.
        """
        if self.adj_pairs is None or self.adj_pairs.size == 0:
            return
        
        adjacency_dict = defaultdict(list)
        adjacency_sets = defaultdict(set)

        for face1, face2 in self.adj_pairs:
            adjacency_dict[face1].append(face2)
            adjacency_dict[face2].append(face1)

            adjacency_sets[face1].add(face2)
            adjacency_sets[face2].add(face1)

        self.adjacency_dict = dict(adjacency_dict)
        self.adjacency_sets = dict(adjacency_sets)

        if len(self.adj_pairs) > 0:
            self.num_faces = np.max(self.adj_pairs) + 1

            self.neighbor_counts = np.zeros(self.num_faces, dtype=np.int32)
            for face_id in range(self.num_faces):
                self.neighbor_counts[face_id] = len(self.adjacency_dict.get(face_id, []))

    def get_first_ring_neighbors(self, face_index):
        """
        Get first ring neighbors.

        Args:
            face_index: index of face to be computed.

        Returns:
            set: neighbors of first ring.
        """
        return self.adjacency_sets.get(face_index, set())

    def get_second_ring_neighbors(self, face_index):
        """
        Get neighbors of second ring with caching.

        Args:
            face_index: index of face

        Returns:
            set: neighbors of second ring.        
        """

        if face_index in self._second_ring_cache:
            return self._second_ring_cache[face_index]
        
        first_ring = self.get_first_ring_neighbors(face_index)

        if not first_ring:
            self._second_ring_cache[face_index] = set()
            return set()
        
        second_ring = set()
        for neighbor in first_ring:
            neighbor_neighbors = self.adjacency_sets.get(neighbor, set())
            second_ring.update(neighbor_neighbors)
        

        second_ring.discard(face_index)
        second_ring -= first_ring

        self._second_ring_cache[face_index] = second_ring

        return second_ring

    def get_neighbor_count(self, face_index):
        """
        Get number of neighbors.

        Args:
            face_index: index of face.
        
        Returns:
            int: number if neighbors.
        """
        if self.neighbor_counts is not None and face_index < len(self.neighbor_counts):
            return self.neighbor_counts[face_index]
        return len(self.adjacency_dict.get(face_index, []))

    def get_all_neighbor_counts(self):
        """
        Get count of all neighbors.

        Returns:
            numpy.ndarray: array with number of neighbors for each face.
        """
        return self.neighbor_counts if self.neighbor_counts is not None else np.array([])

    def batch_get_second_ring_neighbors(self, face_indices):
        """
        Get neighbors of second ring for multiple faces in batch

        Args:
            face_indices: list of face indices.
        
        Returns:
            dict: face_index -> set of a second ring neighbors
        """
        result = {}

        for face_index in face_indices:
            result[face_index] = self.get_second_ring_neighbors(face_index)
        
        return result

    def clear_cache(self):
        """Clean cache of neighbors of second ring"""
        self._second_ring_cache.clear()
