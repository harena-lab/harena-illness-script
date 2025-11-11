"""
This module provides functions to convert between category labels and IDs, 
as well as to generate text representations of category groups.
"""

import random
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

categories_labels = [
    'pathophysiology', 'epidemiology', 'etiology', 'history',
    'physical', 'exams', 'differential', 'therapeutic'
]
cluster_labels = categories_labels + ['total annotations', 'used categories']
# cluster_labels = ['therapeutic', 'exams', 'history', 'total annotations', 'used categories']
N_CATEGORIES = len(categories_labels)

# association between category labels and IDs
categories = {label: idx + 1 for idx, label in enumerate(categories_labels)}

# inverted association between category IDs and labels
inverted_categories = {v: k for k, v in categories.items()}

# Removing cluster methods to be improved
# cluster_methods = [
#     'kmeans 5', 'kmeans n', 'kmeans n cat', 'agglomerative', 'agglomerative cat', 'dbscan',
#     'gmm', 'gmm cat', 'birch', 'hdbscan', 'hdbscan cat', 'spectral'
# ]

cluster_methods = [
    'kmeans emb', 'kmeans 3 emb', 'kmeans cat', 'kmeans 3 cat',
    'agg emb', 'agg 3 emb', 'agg cat', 'agg 3 cat',
    'gmm emb', 'gmm 3 emb', 'gmm cat', 'gmm 3 cat',
    'birch emb', 'birch 3 emb', 'birch cat', 'birch 3 cat'
]

cluster_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

def set_cluster_names(labels):
    global cluster_names
    cluster_names = labels

def categories_labels_to_ids(categories_groups_ids):
    """Converts category labels to their corresponding IDs in a list."""
    cig = []
    for group_id in categories_groups_ids:
        cp = group_id.copy()
        cp[0] = categories[cp[0]]
        cig.append(cp)
    return cig

def categories_ids_to_labels(categories_groups_labels):
    """Converts category IDs to their corresponding labels in a list."""
    cgi = []
    for label_group in categories_groups_labels:
        cp = label_group.copy()
        cp[0] = inverted_categories[cp[0]]
        cgi.append(cp)
    return cgi

def categories_ids_to_text(categories_groups_labels):
    """Generates a text representation of category groups from their IDs."""
    text = ''
    size = len(categories_groups_labels)
    for i in range(size):
        text += (inverted_categories[categories_groups_labels[i][0]] + ':' +
                 str(categories_groups_labels[i][1]) + '/' +
                 str(categories_groups_labels[i][2]))
        if i < size-1:
            text += '; '
    return text

def categories_statistics(categories_ordered):
    """
    Computes statistics about used categories.

    Args:
        categories_ordered (list): A list of category ids with their positions
        and quantity of elements: [[category_id, position, quantity], ...].
        Quantity is optional.

    Returns:
        statistics: A dictionary with the number of elements in each category.
    """
    # compute the number of elements in each category
    statistics = {}
    used_categories = 0
    for cat in range(1, N_CATEGORIES+1):
        count = 0
        for category in categories_ordered:
            if category[0] == cat:
                count += category[2] if len(category) == 3 else 1
        statistics[inverted_categories[cat]] = count
        used_categories += (1 if count > 0 else 0)
    statistics['used categories'] = used_categories

    statistics['total annotations'] = len(categories_ordered)

    ideas = 1
    for ann in range(1, len(categories_ordered)):
        if categories_ordered[ann][1] != categories_ordered[ann-1][1]:
            ideas += 1
    statistics['ideas'] = ideas

    return statistics


# auxiliary function - retrieves the second position
def _order_keys(element):
    return element[1]

def self_order_groups(categories_ordered):
    """
    Orders and groups categories by their adjacent position.

    Args:
        categories_ordered (list): A list of category ids with their positions
        and quantity of elements: [[category_id, position, quantity], ...].
        Quantity is optional.

    Returns:
        list: A list of grouped categories ordered by their position
        [[category_id, position, quantity], ...]
    """
    # sort by text position (second element)
    categories_ordered.sort(key=_order_keys)

    # group by category (first element)
    # group = [category, position of the first group element, score]
    grouped = []
    for cat in range(1, N_CATEGORIES+1):
        prev = -1  # previous in a distinct position (any category)
        prev_g = -1  # previous of the category (any position)
        cat_g = []  # current category grouping
        size = len(categories_ordered)
        for i in range(size):
            if categories_ordered[i][0] == cat:
                quantity = 1 if len(categories_ordered[i]) == 2 else categories_ordered[i][2]
                # if any element in the previous position is not in the same category
                if ((prev == -1 and prev_g == -1) or  # first appearance
                   (prev != -1 and  # has an element in a previous position
                    (categories_ordered[prev][0] != cat and  # not in the same category
                    (prev_g == -1 or  # has no previous element of the category
                     # previous in the category precedes last annotation position
                     categories_ordered[prev][1] > categories_ordered[prev_g][1])))
                   ):
                    cat_g = [cat, categories_ordered[i][1], quantity]  # new category grouping
                    grouped.append(cat_g)
                else:
                    cat_g[2] += quantity
                prev_g = i
            # last distinct position in the sequence (annotations in the same position)
            if (i+1 == len(categories_ordered) or
                categories_ordered[i+1][1] != categories_ordered[i][1]):
                prev = i

    # sort groups by position (second element)
    grouped.sort(key=_order_keys)
    # print(grouped)

    return grouped

def self_order_score(self_order_grouped):
    """
    Calculates the number of substitutions needed to group together categories.

    Args:
        self_order_grouped (list): A list of grouped categories ordered by their position
        [[category_id, quantity], ...]

    Returns:
        int: The number of substitutions needed.
    """
    # score order change to group together categories
    subs = 0
    for cat in range(1, N_CATEGORIES+1):
        prev = -1
        i = 0
        while i < len(self_order_grouped):
            if self_order_grouped[i][0] == cat:
                if prev == -1:
                    prev = i
                else:
                    subs += 1
                    self_order_grouped[prev][2] += self_order_grouped[i][2]
                    self_order_grouped = (self_order_grouped[slice(0, i)] + \
                                          self_order_grouped[slice(i+1, len(self_order_grouped))])
            i += 1

    return subs

def normalized_self_order_score(self_order_grouped):
    """
    Calculates the inverse number of substitutions needed to group together categories, normalized
    by the number of used categories from all available categories.

    Args:
        self_order_grouped (list): A list of grouped categories ordered by their position
        [[category_id, quantity], ...]

    Returns:
        float: The normalized self order score.
    """
    so_score = self_order_score(self_order_grouped)

    # count how many categories were used
    used_categories = 0
    for cat in range(1, N_CATEGORIES+1):
        for category in self_order_grouped:
            if category[0] == cat:
                used_categories += 1
                break

    return 0 if len(self_order_grouped) == 0 else (
           (1 - (so_score / len(self_order_grouped))) * used_categories / N_CATEGORIES)

def clustering_free_recall(categories_ordered):
    """
    Calculates the clustering in free recall for a given order of categories.

    Args:
        categories_ordered (list): A list of category ids with their positions
        [[category_id, position], ...].

    Returns:
        float: The adjusted ratio of clustering.
    """
    n = len(categories_ordered)  # number of recalled items

    # sort by text position (second element)
    categories_ordered.sort(key=_order_keys)

    nc = {}  # number of recalled items in each recalled category
    r = 0  # number of category repetition
    size = len(categories_ordered)
    for i in range(size):
        cat = categories_ordered[i][0]
        if not cat in nc:
            nc[cat] = 1
        else:
            nc[cat] += 1
        next_pos = i + 1
        # find next position of the same category or neighbor start
        while (next_pos < len(categories_ordered) and
               categories_ordered[next_pos][0] != cat and
               categories_ordered[next_pos][1] == categories_ordered[i][1]):
            next_pos += 1
        if next_pos < len(categories_ordered):
            sp = next_pos
            while (sp < len(categories_ordered) and
                   categories_ordered[sp][1] == categories_ordered[next_pos][1]):
                if cat == categories_ordered[sp][0]:
                    r += 1
                    break
                sp += 1

    c = len(nc)  # number of recalled categories
    maximum = n - c  # maximum possible number of category repetitions

    er = 0  # expected number of category repetitions
    for cat, number in nc.items():
        er += number * number
    er = er / n - 1 if n > 1 else None

    # indexes for documentation (not used)
    # rr = r / (n - 1)  # ratio of repetition
    # mrr = r / maximum  # modified ratio of repetition
    # ds = r - er  # deviation score

    # adjusted ratio of clustering
    arc = (('' + str(r - er)) +
           ('/' + str(maximum - er)) if maximum - er == 0 else (r - er) / (maximum - er)
          ) if er is not None else None

    return arc

def find_optimal_k(X, min_k, max_k):
    sil_scores = []
    ch_scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))

    optimal_k_sil = sil_scores.index(max(sil_scores)) + min_k
    optimal_k_ch = ch_scores.index(max(ch_scores)) + min_k

    return max(optimal_k_sil, optimal_k_ch)

# Two alternative methods to find the best eps for DBSCAN - none is working properly

# def find_optimal_eps(X):
#     neigh = NearestNeighbors(n_neighbors=2)
#     nbrs = neigh.fit(X)
#     distances, indices = nbrs.kneighbors(X)
#     distances = np.sort(distances, axis=0)
#     distances = distances[:,1]
#     knee = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
#     return distances[knee.knee]

# def optimize_dbscan_parameters(embeddings_scaled, n_clusters=3, max_unbalance_ratio=0.7):
#     """
#     Find optimal DBSCAN parameters that produce balanced clusters
#     """
#     # Calculate range of possible eps values
#     neighbors = NearestNeighbors(n_neighbors=2)
#     neighbors_fit = neighbors.fit(embeddings_scaled)
#     distances, _ = neighbors_fit.kneighbors(embeddings_scaled)
#     distances = np.sort(distances[:, 1])
    
#     # Try different eps values
#     possible_eps = np.linspace(np.percentile(distances, 1), 
#                              np.percentile(distances, 90), 
#                              num=50)
    
#     best_params = None
#     best_score = float('inf')
    
#     for eps in possible_eps:
#         # Try different min_samples values
#         for min_samples in range(2, int(len(embeddings_scaled) * 0.1)):
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             labels = dbscan.fit_predict(embeddings_scaled)
            
#             # Skip if we have noise points (-1 label)
#             if -1 in labels:
#                 continue
                
#             unique_labels = np.unique(labels)
#             n_clusters_found = len(unique_labels)
            
#             # Skip if we don't have desired number of clusters
#             if n_clusters_found != n_clusters:
#                 continue
            
#             # Calculate cluster sizes
#             cluster_sizes = np.bincount(labels)
#             max_size = np.max(cluster_sizes)
#             min_size = np.min(cluster_sizes)
            
#             # Calculate balance score (lower is better)
#             balance_score = max_size / min_size
            
#             # Update best parameters if we found better balance
#             if balance_score < best_score and balance_score <= 1/max_unbalance_ratio:
#                 best_score = balance_score
#                 best_params = (eps, min_samples)
    
#     if best_params is None:
#         raise ValueError("Could not find parameters for balanced clustering")
        
#     return best_params

def clustered_scores(self_order_scores):
    """
    Clusters sentences based on their embeddings and calculates statistics for each cluster.
    Add columns with cluster data to the DataFrame.
    
    Args:
        self_order_scores (DataFrame): A DataFrame with a list of grouped categories ordered
        by their position and sentence scores.

    Returns:
        DataFrame: cluster statistics.
    """

    id_column = 'annotation id'
    self_order_column = 'self order grouped'
    additional_columns = ['objective test score', 'organization level', 'global score']

    # check if the specified columns exist
    required_columns = [id_column] + [self_order_column] + additional_columns
    missing_columns = [col for col in required_columns if col not in self_order_scores.columns]
    if missing_columns:
        raise ValueError(
            f"Columns {missing_columns} not found in the CSV. " +
            f"Available columns are: {self_order_scores.columns.tolist()}")

    # Apply categories_statistics to the 'self order grouped' column
    self_order_scores['category statistics'] = (
        self_order_scores['categories ordered'].apply(categories_statistics))

    # Split the dictionary returned by categories_statistics into separate columns
    category_stats_df = pd.json_normalize(self_order_scores['category statistics'])
    for col in category_stats_df.columns:
        self_order_scores[col] = category_stats_df[col]

    additional_columns = categories_labels + additional_columns

    self_order_scores['computed score'] = (
        (self_order_scores['objective test score'] / 10) + self_order_scores['global score']) / 2

    # generate embeddings for each sentence
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    self_order_sentences = self_order_scores[self_order_column].tolist()
    embeddings = model.encode(self_order_sentences)
    embeddings = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # generate category-based dimensions for clustering
    # category_dimensions = self_order_scores[categories_labels].values
    # category_sum = category_dimensions.sum(axis=1, keepdims=True)
    # category_dimensions = category_dimensions / category_sum
    # category_dimensions = np.hstack((category_dimensions, self_order_scores['total annotations'].values[:, np.newaxis]))
    # category_dimensions = np.hstack((category_dimensions, self_order_scores['normalized self order score'].values[:, np.newaxis]))
    category_dimensions = self_order_scores[cluster_labels].values

    # category_dimensions = self_order_scores[['total annotations', 'normalized self order score']].values   
    
    # category_dimensions = category_dimensions * (
    #     self_order_scores['normalized self order score'].values[:, np.newaxis])
    category_dimensions = np.nan_to_num(category_dimensions)
    
    # categories_scaled = scaler.fit_transform(category_dimensions)
    categories_scaled = category_dimensions

    cluster_summary = []
    min_clusters = 3
    max_clusters = 20

    optimal_k_emb = find_optimal_k(embeddings_scaled, min_clusters, max_clusters)
    kmeans_emb = KMeans(n_clusters=optimal_k_emb,
                      n_init=10,
                      max_iter=300,
                      init='k-means++',
                      random_state=42)
    self_order_scores['cluster kmeans emb'] = kmeans_emb.fit_predict(embeddings_scaled)
    cluster_summary.append(evaluate_clusters(
        'kmeans emb', embeddings_scaled, self_order_scores['cluster kmeans emb']))

    kmeans_3_emb = KMeans(n_clusters=3,
                          n_init=10,
                          max_iter=300,
                          init='k-means++',
                          random_state=42)
    self_order_scores['cluster kmeans 3 emb'] = kmeans_3_emb.fit_predict(embeddings_scaled)
    cluster_summary.append(evaluate_clusters(
        'kmeans 3 emb', embeddings_scaled, self_order_scores['cluster kmeans 3 emb']))

    optimal_k_cat = find_optimal_k(categories_scaled, min_clusters, max_clusters)
    kmeans_cat = KMeans(n_clusters=optimal_k_cat,
                        n_init=10,
                        max_iter=300,
                        init='k-means++',
                        random_state=42)
    self_order_scores['cluster kmeans cat'] = kmeans_cat.fit_predict(categories_scaled)
    cluster_summary.append(evaluate_clusters(
        'kmeans cat', categories_scaled, self_order_scores['cluster kmeans cat']))

    kmeans_3_cat = KMeans(n_clusters=3,
                          n_init=10,
                          max_iter=300,
                          init='k-means++',
                          random_state=42)
    self_order_scores['cluster kmeans 3 cat'] = kmeans_3_cat.fit_predict(categories_scaled)
    cluster_summary.append(evaluate_clusters(
        'kmeans 3 cat', categories_scaled, self_order_scores['cluster kmeans 3 cat']))

    # Agglomerative Clustering (uses unscaled data)
    agg_emb = AgglomerativeClustering(n_clusters=optimal_k_emb)
    self_order_scores['cluster agg emb'] = agg_emb.fit_predict(embeddings)
    cluster_summary.append(evaluate_clusters(
        'agg emb', embeddings, self_order_scores['cluster agg emb']))

    agg_3_emb = AgglomerativeClustering(n_clusters=3)
    self_order_scores['cluster agg 3 emb'] = agg_3_emb.fit_predict(embeddings)
    cluster_summary.append(evaluate_clusters(
        'agg 3 emb', embeddings, self_order_scores['cluster agg 3 emb']))

    agg_cat = AgglomerativeClustering(n_clusters=optimal_k_cat)
    self_order_scores['cluster agg cat'] = agg_cat.fit_predict(category_dimensions)
    cluster_summary.append(evaluate_clusters(
        'agg cat', category_dimensions, self_order_scores['cluster agg cat']))

    agg_3_cat = AgglomerativeClustering(n_clusters=3)
    self_order_scores['cluster agg 3 cat'] = agg_3_cat.fit_predict(category_dimensions)
    cluster_summary.append(evaluate_clusters(
        'agg 3 cat', category_dimensions, self_order_scores['cluster agg 3 cat']))

    # DBSCAN (uses scaled data)
    # (temporarily disabled to improve the method)
    # --- method 1 - unbalanced clusters
    # optimal_eps = find_optimal_eps(embeddings_scaled)
    # dbscan = DBSCAN(eps=optimal_eps, min_samples=min_clusters-1)
    # --- method 2 - error in balance
    # eps, min_samples = optimize_dbscan_parameters(embeddings_scaled, n_clusters=3)
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # self_order_scores['cluster dbscan'] = dbscan.fit_predict(embeddings_scaled)
    # cluster_summary.append(evaluate_clusters(
    #     'dbscan', embeddings_scaled, self_order_scores['cluster dbscan']))

    # Gaussian Mixture Model (uses scaled data)
    gmm_emb = GaussianMixture(n_components=optimal_k_emb, random_state=42)
    self_order_scores['cluster gmm emb'] = gmm_emb.fit_predict(embeddings_scaled)
    cluster_summary.append(evaluate_clusters(
        'gmm emb', embeddings_scaled, self_order_scores['cluster gmm emb']))

    gmm_3_emb = GaussianMixture(n_components=3, random_state=42)
    self_order_scores['cluster gmm 3 emb'] = gmm_3_emb.fit_predict(embeddings_scaled)
    cluster_summary.append(evaluate_clusters(
        'gmm 3 emb', embeddings_scaled, self_order_scores['cluster gmm 3 emb']))
    
    gmm_cat = GaussianMixture(n_components=optimal_k_cat, random_state=42)
    self_order_scores['cluster gmm cat'] = gmm_cat.fit_predict(categories_scaled)
    cluster_summary.append(evaluate_clusters(
        'gmm cat', categories_scaled, self_order_scores['cluster gmm cat']))

    gmm_3_cat = GaussianMixture(n_components=3, random_state=42)
    self_order_scores['cluster gmm 3 cat'] = gmm_3_cat.fit_predict(categories_scaled)
    cluster_summary.append(evaluate_clusters(
        'gmm 3 cat', categories_scaled, self_order_scores['cluster gmm 3 cat']))

    # Birch (uses unscaled data, but can benefit from scaling in some cases)
    birch_emb = Birch(n_clusters=optimal_k_emb, threshold=0.01)
    self_order_scores['cluster birch emb'] = birch_emb.fit_predict(embeddings)
    cluster_summary.append(evaluate_clusters(
        'birch emb', embeddings, self_order_scores['cluster birch emb']))

    birch_3_emb = Birch(n_clusters=3, threshold=0.01)
    self_order_scores['cluster birch 3 emb'] = birch_3_emb.fit_predict(embeddings)
    cluster_summary.append(evaluate_clusters(
        'birch 3 emb', embeddings, self_order_scores['cluster birch 3 emb']))

    birch_cat = Birch(n_clusters=optimal_k_cat, threshold=0.01)
    self_order_scores['cluster birch cat'] = birch_cat.fit_predict(embeddings)
    cluster_summary.append(evaluate_clusters(
        'birch cat', embeddings, self_order_scores['cluster birch cat']))

    birch_3_cat = Birch(n_clusters=3, threshold=0.01)
    self_order_scores['cluster birch 3 cat'] = birch_3_cat.fit_predict(embeddings)
    cluster_summary.append(evaluate_clusters(
        'birch 3 cat', embeddings, self_order_scores['cluster birch 3 cat']))

    # HDBSCAN (uses unscaled data)
    # (temporarily disabled to improve the method)
    # hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    # self_order_scores['cluster hdbscan'] = hdbscan_clusterer.fit_predict(embeddings)
    # cluster_summary.append(evaluate_clusters(
    #     'hdbscan', embeddings, self_order_scores['cluster hdbscan']))
    
    # self_order_scores['cluster hdbscan cat'] = hdbscan_clusterer.fit_predict(categories_scaled)
    # cluster_summary.append(evaluate_clusters(
    #     'hdbscan cat', category_dimensions, self_order_scores['cluster hdbscan cat']))

    # Spectral Clustering (uses scaled data)
    # (temporarily disabled to improve the method)
    # spectral = SpectralClustering(n_clusters=optimal_k, random_state=42)
    # self_order_scores['cluster spectral'] = spectral.fit_predict(embeddings_scaled)
    # cluster_summary.append(evaluate_clusters(
    #     'spectral', embeddings_scaled, self_order_scores['cluster spectral']))

    # calculate mean and standard deviation for each additional column within each cluster
    cluster_stats = []
    for cm in cluster_methods:
        for cluster in range(len(np.unique(self_order_scores[f'cluster {cm}']))):
            cluster_data = self_order_scores[self_order_scores[f'cluster {cm}'] == cluster]
            cluster_data = cluster_data.reset_index(drop=True)
            stats = {'method': cm, 'cluster': cluster, 'size': len(cluster_data)}
            for col in additional_columns + ['computed score']:
                if not cluster_data.empty:
                    stats[f'{col}_min'] = cluster_data[col].min()
                    stats[f'{col}_max'] = cluster_data[col].max()
                    stats[f'{col}_mean'] = (
                        cluster_data[col].mean() if len(cluster_data) > 1
                            else cluster_data[col].iloc[0])
                    stats[f'{col}_std'] = (
                        cluster_data[col].std() if len(cluster_data) > 1 else 0)
                    stats[f'{col}_median'] = (
                        cluster_data[col].median() if len(cluster_data) > 1
                            else cluster_data[col].iloc[0])
                else:
                    stats[f'{col}_mean'] = 0
                    stats[f'{col}_std'] = 0
                    stats[f'{col}_median'] = 0
            sum = 0
            for cl in categories_labels:
                sum += stats[f'{cl}_mean']
            for cl in categories_labels:
                stats[f'{cl}_norm_mean'] = '' if sum == 0 else stats[f'{cl}_mean'] / sum
            cluster_stats.append(stats)

    # create a DataFrame with cluster statistics
    cluster_stats_df = pd.DataFrame(cluster_stats)

    # Technique deprecated, possibly must be simplified
    #
    # # Perform hierarchical clustering on the mean computed scores
    # mean_scores = cluster_stats_df['computed score_median'].values.reshape(-1, 1)
    # Z = linkage(mean_scores, method='ward')

    # # Assign cluster labels
    # cluster_stats_df['hierarchical cluster'] = fcluster(Z, t=3, criterion='maxclust')

    # fs = {}
    # for _, cs in cluster_stats_df.iterrows():
    #     hc = cs['hierarchical cluster']
    #     if hc not in fs or cs['computed score_median'] > fs[hc]:
    #         fs[hc] = cs['computed score_median']

    # # transform fs in a list order by value
    # fs = sorted(fs.items(), key=lambda x: x[1])

    # # replace the value by an ordered number
    # h_cluster_map = {}
    # for i in fs:
    #     h_cluster_map[i[0]] = len(h_cluster_map) + 1
    # # map in  a new column of the DataFrame
    # cluster_stats_df['estimated grade'] = (
    #     cluster_stats_df['hierarchical cluster'].map(h_cluster_map))

    # # transform the columns cluster and estimated grade in a map
    # cluster_map = {}
    # for i in range(5):
    #     cluster_map[i] = (
    #         cluster_stats_df[cluster_stats_df['cluster'] == i]['estimated grade'].values[0])

    # # map back to the self_order_scores the estimated grade according to the cluster
    # self_order_scores['estimated grade'] = (
    #     self_order_scores['cluster kmeans 3 emb'].map(cluster_map))

    return {
        'stats': cluster_stats_df,
        'summary': pd.DataFrame(cluster_summary)
    }

def evaluate_clusters(method, embeddings, clustering_results):
    n_clusters = len(np.unique(clustering_results))
    return {
        'method': method,
        'n clusters': n_clusters,
        'silhouette': '' if (n_clusters <= 1) else silhouette_score(embeddings, clustering_results),
        'calinski harabasz': '' if (n_clusters <= 1) else calinski_harabasz_score(embeddings, clustering_results)
    }

# transform the field 'categories ordered' of items as 'category:position/quantity; category:position/quantity; ...'
# into [[category, position, quantity], [category, position, quantity], ...]
def parse_categories_ordered(x):
    items = x.split(';')
    result = []
    for item in items:
        parts = item.split(':')
        if len(parts) == 2:
            category = parts[0].strip()
            pos_qty = parts[1].split('/')
            if len(pos_qty) == 2:
                position = int(pos_qty[0])
                quantity = int(pos_qty[1])
                result.append([category, position, quantity])
    return result

def annotation_metrics(categories_order_csv, sentence_scores_csv,
                       annotation_metrics_csv, annotations_summary_csv, annotation_stats_csv,
                       self_order_grouped_input=False, medical_specialist_year='resultados_anotacoes_teste_progresso_dpoc.csv'):
    """
    Generates metrics for annotations based on the order of categories and sentence scores.

    Args:
        categories_order_csv (str): Path to the CSV file with, for each sentence, a list of
          categories with their positions and quantity of elements:
          'category: position/quantity; ....'
        sentence_scores_csv (str): Path to the CSV file with sentence scores
        annotation_metrics_csv (str): Path to the output CSV file with annotation metrics.
        annotation_stats_csv (str): Path to the output CSV file with annotation statistics.
    """
    annotations = pd.read_csv(categories_order_csv)

    # Fill NaN values with an empty string
    annotations['categories ordered'] = annotations['categories ordered'].fillna('')

    annotations['categories ordered'] = (
        annotations['categories ordered'].apply(parse_categories_ordered))

    # convert category labels into numbers
    annotations['categories ordered'] = (
        annotations['categories ordered'].apply(categories_labels_to_ids))

    if self_order_grouped_input:
        annotations['self order grouped'] = annotations['categories ordered']
    else:
        # aggregate as self order grouped
        annotations['self order grouped'] = (
            annotations['categories ordered'].apply(self_order_groups))

    annotations['self order score'] = (
        annotations['self order grouped'].apply(self_order_score))

    annotations['normalized self order score'] = (
        annotations['self order grouped'].apply(normalized_self_order_score))

    annotations['clustering free recall'] = (
        annotations['categories ordered'].apply(clustering_free_recall))

    # convert self order grouped into a string
    annotations['self order grouped'] = (
        annotations['self order grouped'].apply(categories_ids_to_text))

    sentence_scores = pd.read_csv(sentence_scores_csv)
    annotations = pd.merge(sentence_scores, annotations, on='annotation id', how='inner')

    cluster_results = clustered_scores(annotations)

    # write to a csv file
    # annotations.drop(
    #     columns=['categories ordered','self order grouped','category statistics'],
    #     inplace=True)

    """
    GENERATING YEAR_MEAN AND YEAR_STD for by method and cluster. 
    Inserting year to metrics dataframe
    Insterting year_mean and year_std in stats dataframe
    """
    student_year = pd.read_csv('resultados_anotacoes_teste_progresso_dpoc.csv')

    student_year = student_year[['annotation id', 'year or semester','temporality']]
    student_year['year or semester'] = student_year.apply(
        lambda row: max(1, row['year or semester'] / 2) if row['temporality'].lower() == 'semestral' else row['year or semester'],
        axis=1
    )
    student_year.rename(columns={'year or semester': 'year'}, inplace=True)
    student_year.drop(columns=['temporality'], inplace=True)

    year_start = int(student_year['year'].min())
    year_end = int(student_year['year'].max())

    annotations = pd.merge(annotations, student_year, on='annotation id')

    # Extract columns starting with 'cluster' from annotations
    cluster_columns = [col for col in annotations.columns if col.startswith('cluster')]

    # Create a dictionary to hold dataframes for each method
    method_dfs = {}

    # Separate each method into its own dataframe
    for col in cluster_columns:
        method_name = col.split('cluster ')[1] if 'cluster ' in col else None  # Assuming the format is 'cluster method'
        method_df = annotations[['annotation id', 'year', col]].rename(columns={col: 'cluster'})
        method_df['method'] = method_name
        method_dfs[method_name] = method_df

    # Combine all method dataframes into a single dataframe
    df_clusters_methods = pd.concat(method_dfs.values(), ignore_index=True)

    # Calculate the mean and standard deviation of the 'year' for each cluster and method
    year_stats = df_clusters_methods.groupby(['method', 'cluster'])['year'].agg(
                               ['min', 'max', 'mean', 'std', 'median']).reset_index()
    year_stats.rename(columns={'min': 'year_min', 'max': 'year_max', 'mean': 'year_mean',
                               'std': 'year_std', 'median': 'year_median'}, inplace=True)
    
    # Merge the year_stats back into the original dataframe
    df_clusters_methods = pd.merge(df_clusters_methods, year_stats, on=['method', 'cluster'])
    cluster_results['stats'] = pd.merge(cluster_results['stats'], year_stats, on=['method', 'cluster'], how='left')

    # Create a column for each year (from min to max) and fill with the student's count
    year_columns = []
    for year in range(year_start, year_end + 1):
        year_columns.append(f'year_{year}')
        df_clusters_methods[f'year_{year}'] = df_clusters_methods.apply(
            lambda row, year=year: 1 if row['year'] == year else 0, axis=1)

    # Aggregate the counts for each year by method and cluster
    year_columns = ['method', 'cluster'] + year_columns
    year_counts = df_clusters_methods[year_columns].groupby(['method', 'cluster']).sum().reset_index()

    # Merge the year_counts back into the original dataframe
    df_clusters_methods = pd.merge(df_clusters_methods, year_counts, on=['method', 'cluster'])
    cluster_results['stats'] = pd.merge(cluster_results['stats'], year_counts, on=['method', 'cluster'], how='left')

    # Reorder the lines of the dataframe according to the fields "method" and "year_mean"
    cluster_results['stats'] = cluster_results['stats'].sort_values(by=['method', 'year_mean'])

    # Create a column 'cluster_label' in the dataframe cluster_results['stats']
    cluster_results['stats']['cluster_name'] = ''

    # Iterate over each method and assign cluster labels
    num_cluster_names = len(cluster_names)
    for method in cluster_methods:
        method_indices = cluster_results['stats'][cluster_results['stats']['method'] == method].index
        map_cluster_name = {}
        for i, idx in enumerate(method_indices):
            map_cluster_name[cluster_results['stats'].at[idx, 'cluster']] = i
            cluster_results['stats'].at[idx, 'cluster'] = i
        for idx in annotations.index:
            annotations.at[idx, f'cluster {method}'] = map_cluster_name[annotations.at[idx, f'cluster {method}']]

    annotations = annotations.sort_values('annotation id')
    annotations.to_csv(annotation_metrics_csv, index=False)
    cluster_results['summary'].to_csv(annotations_summary_csv, index=False)
    cluster_results['stats'].to_csv(annotation_stats_csv, index=False)

def compare_clusters(annotation_metrics_csv_1, annotation_metrics_csv_2,
                     annotation_summary_csv_1, annotation_summary_csv_2,
                     annotation_comparison_csv,
                     columns_compare=cluster_methods):
    """
    Compares two clustering of annotations.

    Args:
        annotation_metrics_csv_1 (str): Path to the first CSV file with annotation metrics.
        annotation_metrics_csv_2 (str): Path to the second CSV file with annotation metrics.
        column_compare (str): The column to use for comparison.

    Returns:
        tuple: A tuple with the adjusted rand index and normalized mutual information.
    """
    # Read the CSV files
    dfm1 = pd.read_csv(annotation_metrics_csv_1)
    dfm2 = pd.read_csv(annotation_metrics_csv_2)

    # Ensure that both dataframes have the same annotations and are sorted
    dfm1 = dfm1.sort_values('annotation id')
    dfm2 = dfm2.sort_values('annotation id')

    # Keep only annotations that appear in both files
    common_ids = dfm1['annotation id'].isin(dfm2['annotation id'])
    dfm1 = dfm1[common_ids]
    dfm2 = dfm2[common_ids]

    comparison = []

    for cc in columns_compare:
        column_name = f'cluster {cc}'

        if column_name not in dfm1.columns or column_name not in dfm2.columns:
            raise ValueError(f"Column {column_name} not found in the CSV. " +
                             f"Available columns are: {dfm2.columns.tolist()}")

        # Extract cluster assignments
        clusters1 = dfm1[column_name].values
        clusters2 = dfm2[column_name].values

        # Calculate metrics
        comparison.append({
            'method': cc,
            'ari': adjusted_rand_score(clusters1, clusters2),
            'nmi': adjusted_mutual_info_score(clusters1, clusters2),
            'fm': fowlkes_mallows_score(clusters1, clusters2)
        })

    comparison_df = pd.DataFrame(comparison)

    # Read the annotations summary
    dfs1 = pd.read_csv(annotation_summary_csv_1)
    dfs2 = pd.read_csv(annotation_summary_csv_2)

    # add a sufix _1 in the dfs1 columns and _2 in dfs2 columns
    dfs1.columns = [f'{col}_1' for col in dfs1.columns]
    dfs2.columns = [f'{col}_2' for col in dfs2.columns]

    # Merge comparison_df with dfs1 and dfs2
    comparison_df = comparison_df.merge(
        dfs1, left_on='method', right_on='method_1').drop(columns=['method_1'])
    comparison_df = comparison_df.merge(
        dfs2, left_on='method', right_on='method_2').drop(columns=['method_2'])

    comparison_df.to_csv(annotation_comparison_csv, index=False)
