import numpy as np
import faiss # Fast Approximate Nearest‑Neighbor search
import networkx as nx
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import threading


def select_central_nodes(graph, centrality_dict, top_n=5, min_distance=2):
    """Return top_n central nodes while enforcing a minimum hop distance.

    This function performs a greedy selection: at each iteration we pick the
    currently highest-ranked node whose shortest-path distance to every node
    already in selected_nodes is at least min_distance.

    Parameters
    ----------
    graph : nx.Graph
        Graph on which centrality was computed.
    centrality_dict : dict
        Pre-computed centrality scores (e.g. PageRank).
    top_n : int, default=5
        Desired number of nodes to return.
    min_distance : int, default=2
        Minimum distance (in hops) allowed between any pair of returned nodes.

    Returns
    -------
    list[tuple[node, float]]
        List of *(node, score)* tuples which our the nodes we will use to represent a user or genre.
    """
    selected_nodes = []
    centrality_scores = centrality_dict.copy()
    
    for _ in range(top_n):
        max_node = None
        max_score = -1
        
        for node, score in centrality_scores.items():
            # Skip candidates too close to already‑selected nodes
            if all(nx.shortest_path_length(graph, source=node, target=sel_node) >= min_distance for sel_node, sel_score in selected_nodes):
                if score > max_score:
                    max_node = node
                    max_score = score
        
        if max_node is not None:
            selected_nodes.append((max_node, max_score))
            del centrality_scores[max_node]
        else:
            break  # No more valid nodes to select

    # If we ran out of valid nodes early, pad with highest remaining    
    if len(selected_nodes) < top_n:
        sorted_centralities = sorted(centrality_scores.items(), key=lambda item: item[1], reverse=True)
        selected_nodes.extend(sorted_centralities[:top_n-len(selected_nodes)])
            
        
    return selected_nodes

def get_group_hubs(group, tn=5):
    """Identify hub items inside group using FAISS + PageRank.

    Steps
    -----
    1. Use FAISS index on embeddings to obtain k-NN edges.
    2. Construct an undirected similarity graph and run PageRank.
    3. Call select_central_nodes on nodes+PageRank scores to pick central nodes that enforce maximum diversity.

    Parameters
    ----------
    group : Sequence[SongObject]
        Collection providing embeddings (length-d vectors).
    tn : int, default=5
        Number of hub nodes to return.

    Returns
    -------
    list[tuple[int, float]]
        Output of select_central_nodes - node indices and scores.

    """
    embeddings = np.array([np.array(song.embedding) for song in group])
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    embeddings = embeddings.astype(np.float32)
    index.add(embeddings)

    k = 10  # Number of nearest neighbors to find
    distances, indices = index.search(embeddings, k)

    G = nx.Graph()
    
    for i in range(len(indices)):
        for j in indices[i]:
            if i != j:
                G.add_edge(i, j)

    pagerank_scores = nx.pagerank(G)
    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    top_nodes = select_central_nodes(G, pagerank_scores, top_n=tn, min_distance=2)

    print(f"Top {tn} nodes by PageRank:")
    for node, score in top_nodes:
        print(f"Node {node}, PageRank: {score}")

    return top_nodes

def bfs(graph, start_node, end_node):
    """Breadth-first search that returns the first found path.

    Parameters
    ----------
    graph : nx.Graph
        Graph representation.
    start_node, end_node : Hashable
        Ids of the start and goal nodes.

    Returns
    -------
    list | None
        A list of node IDs representing the path (inclusive) or *None* if no
        path exists.
    """

    
    visited = set()  # Track visited nodes during this BFS search
    queue = deque([[start_node]])  # Queue to store paths

    if start_node == end_node:
        return [start_node]

    while queue:
        path = queue.popleft()  # Get the next path to explore
        node = path[-1]  # Get the last node in the path

        if node not in visited:
            neighbors = graph[node]

            for neighbor in neighbors:
                # Only visit nodes that are not used and not already visited
                # if neighbor not in visited and neighbor not in used_nodes:
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

                    # If the neighbor is the end node, we found a valid path
                    if neighbor == end_node:
                        return new_path

            # Mark the current node as visited
            visited.add(node)

    # If no path is found, return None
    return None

def generate_paths(graph, user_hubs, target_hubs):
    """Generate candidate paths between user and target hub sets.

    Parameters
    ----------
    graph : nx.Graph 
        Full k-NN graph.
    user_hubs, target_hubs : Sequence[SongObject]
        User and target node objects exposing song_id.

    Returns
    -------
    list[list]
        Each inner list is a sequence of song_id values representing a path in the graph.
    """
     
    all_paths = []

    for start_item in [liked_song.song_id for liked_song in user_hubs]:
        for end_item in [song.song_id for song in target_hubs]:
            path = bfs(graph, start_item, end_item)
            if path is not None:
                all_paths.append(path)
            
    return all_paths

def standardize_path(path, target_length, graph, path_set):
    """Resize path to target_length via interpolation or pruning.

    Parameters
    ----------
    path : list
        Original path as a list of node IDs.
    target_length : int
        Desired number of steps in the returned path.
    graph : nx.Graph
        Graph providing neighborhood information.
    path_set : Iterable[SongObject]
        Universe from which embeddings are fetched.

    Returns
    -------
    list
        A new path with exactly target_length elements.
    """
    current_length = len(path)

    path_embeddings = {song.song_id: np.array(song.embedding) for song in path_set}
    
    if current_length == target_length:
        return path
    
    standardized_path = list(path)
    
    # If the path is shorter, interpolate
    if current_length < target_length:
        while len(standardized_path) < target_length:
            for i in range(len(standardized_path) - 1):
                if len(standardized_path) < target_length:
                    # Find a neighbor to insert between node i and i+1
                    neighbors = graph[standardized_path[i]]
                    next_node = standardized_path[i + 1]
                    
                    # Insert a neighbor that is closer to next_node
                    for neighbor in neighbors:
                        if neighbor != next_node and neighbor not in standardized_path:
                            standardized_path.insert(i + 1, neighbor)
                            break

    # If the path is longer, prune
    elif current_length > target_length:
        while len(standardized_path) > target_length:
            cosine_similarities = []
            for i in range(1, len(standardized_path) - 1):  # Exclude the first and last nodes
                prev_node = standardized_path[i - 1]
                next_node = standardized_path[i + 1]
                
                # Get embeddings for the previous and next nodes
                prev_embedding = path_embeddings[prev_node].reshape(1, -1)
                next_embedding = path_embeddings[next_node].reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(prev_embedding, next_embedding)[0][0]
                cosine_similarities.append((i, similarity))
            
            # Find the node index with the highest cosine similarity between its previous and next nodes
            if cosine_similarities:
                prune_index, _ = max(cosine_similarities, key=lambda x: x[1])
                standardized_path.pop(prune_index)
            else:
                break  # No nodes left to prune

    return standardized_path

def sample_paths(paths, step):
    """Sample items from paths at the given step.

    Parameters
    ----------
    paths : Sequence[PathObject]
        Sequence of path objects that implement the function get_success_rate_from_current_distribution and
        and connect to all song objects in the path.
    step : int
        Current step in the paths to sample from.

    Returns
    -------
    recos : list
        List of 20 recommended songs as IDs.
    top_paths : list
        List of all sampled paths indexed until the current step
    """

    #get a sample from each posterior
    post_samps = [path.get_success_rate_from_current_distribution() for path in paths]
    post_samps = np.array(post_samps)

    sorted_samps = np.argsort(post_samps)

    recos = []
    top_paths = []
    
    while len(recos) < 20:
        top_idx, sorted_samps = sorted_samps[0], sorted_samps[1:]
        path_songs = paths[top_idx].pathsong_set.all()
        songs_in_order = [ps.song.song_id for ps in path_songs]
        top_song = songs_in_order[step-1]

        top_paths.append(songs_in_order[:step])
        if top_song not in recos:
            recos.append(top_song)
    
    return recos, top_paths

def update_paths(paths, step, sampled_songs, tau=1, epsilon=0):
    """Update path distributions with feedback from sampled_songs. Propagate to non-sampled paths.

    Parameters
    ----------
    paths : Sequence[PathObject]
        Paths to update distribution from.
    step : int
        Index of the current interaction step.
    sampled_songs : Mapping[str, float]
        Map of song_id → reward.
    tau : float, default=1.0
        Temperature parameter for similarity scaling.
    epsilon : float, default=0.0
        Minimum cosine similarity considered for cross-path updates.

    Returns
    -------
    Sequence[PathObject]
        The same list with updated utility belief distributions, mutated in place for convenience.
    """
        
    current_step_songs = []
    for path in paths:
        path_songs = path.pathsong_set.all()
        songs_in_order = [ps.song.song_id for ps in path_songs]
        current_step_songs.append(songs_in_order[step-1])

    # Aggregate embeddings for similarity calculations
    all_path_songs = list(itertools.chain.from_iterable([path.songs.all() for path in paths]))
    path_embeddings = {song.song_id: np.array(song.embedding) for song in all_path_songs}

    for i in range(len(paths)):
        song = current_step_songs[i]
        if song in sampled_songs.keys():
            score = sampled_songs[song]

            paths[i].update_current_distribution(score)

            song_embedding = path_embeddings[song].reshape(1, -1)

            for j in range(len(paths)):
                if i == j:
                    continue
                song2 = current_step_songs[j]
                song2_embedding = path_embeddings[song2].reshape(1, -1)
                similarity = cosine_similarity(song_embedding, song2_embedding)[0][0]
    
                if similarity > epsilon:
                    paths[j].update_current_distribution(score, similarity * 1/tau)
    return paths
