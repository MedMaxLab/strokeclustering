# NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import linalg
from scipy.ndimage import center_of_mass
from sklearn import cluster
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import networkx as nx
from plotly.offline import plot
import plotly.graph_objects as go

##########################
###       IMAGES       ###
##########################
def center_of_mass_coord(mask):
    mask_array = mask.get_fdata()
    cf = np.array([90, -126, -72])
    cm = np.array(center_of_mass(mask_array))
    cm[0] = -cm[0]
    cut_coords = cm + cf
    return cut_coords
    
###########################
### CORRELATION STUDIES ###
###########################

# PERMUTATION ESTIMATE
def permutation_pval_estimate(var_df, n_permutations=1e4, verbose=False):
    '''
    This function computes p-values for the spearman correlations, as suggested for datasets smaller than ~500 subjects.
    
    Parameters:
    var_df (pandas.DataFrame): table of data (n_subjects, n_features), on which correlations and p-values are estimated.
    n_permutations (int): number of permutations to estimate p-values.
    verbose (boolean): whether to print current iteration.

    Returns:
    est_pval_arr (numpy.ndarray): 2-D array containing p-values.
    '''
    est_pval_arr = np.empty(shape=(len(var_df.columns), len(var_df.columns)))
    estimate = 1
    all_estimates = len(var_df.columns)*(len(var_df.columns) - 1) // 2
    for i in range(len(var_df.columns)): # do not compute p-values for autocorrelations and double pairs
        for j in range(i):
            if i==j:
                est_pval_arr[i,j] = 0.0
                continue
            var_x = var_df.iloc[:,i]
            var_y = var_df.iloc[:,j]
            def statistic(var_x):  # permute only x
                return scipy.stats.spearmanr(var_x, var_y).correlation
            res_exact = scipy.stats.permutation_test((var_x,), statistic,
                                           permutation_type='pairings', n_resamples=n_permutations)
            est_pval_arr[i,j] = res_exact.pvalue
            if verbose:
                print("Estimate of hypothesis #{} over {} complete".format(estimate, all_estimates))
                estimate+=1
    est_pval_arr = est_pval_arr + est_pval_arr.T
    return est_pval_arr

# BENJAMINI-HOCHBERG CORRECTION
def hochberg_correction(pvalmat, alpha=0.05):
    '''
    This function applies the Benjamini-Hochberg step-up correction for multiple comparison:
    the p-values are ranked and each one has a specifically corrected threshold, given a significance value.
    This thresholds are the Relative Upper Bounds (RUB) of the various p-values.
    The formula for the RUB of the k-th p-value, at alpha significance, is:
    alpha/(m-k)
    The last p-value below its RUB is the last p-value for the rejection of its H0.
    Be its rank K, then reject all the first K null hypotheses.

    Parameters:
    pvalmat (array-like): complete matrix of estimated p-values for pairwise correlations.
    alpha (float): desired significance level cut-off.
    
    Returns:
    ranked_pvals (array-like): 1-D array of sorted p-values
    rankmax (int): rank, index of the last significant p-value after correction. Add one to obtain the counts of significant correlations.
    '''
    pval_flat = pvalmat[np.tril_indices_from(pvalmat, k=-1)] # pvalues for autocorrelations (on diagonal) do not matter
    
    # Start by ordering the m p-values (from lowest to highest), and relative null hypotheses
    ranked_pvals = np.sort(pval_flat)
    m = len(ranked_pvals)
    
    rankmax=0
    for k,pval in enumerate(ranked_pvals):
        if pval <= ( alpha / (m - k + 1)): # k is 0-based in python, hence use k+1
            rankmax=k # update rank of maximum pval below its rub
    # end for-cycle
    
    # reject the null hypotheses associated to all p-values lesser than R-th p-value
    return ranked_pvals, rankmax

##########################
### PROXIMITY MEASURES ###
##########################

# GDM
def dab(a,b,X):
    '''
    Calculate the distance between two data points.

    Parameters:
    a (int): Index of the first data point.
    b (int): Index of the second data point.
    X (numpy.ndarray): The dataset containing data points.

    Returns:
    dab (float): The Generalized Distance (dab) between data points Xa and Xb.

    Note:
    - The dab distance is computed based on three terms:
      1. The first term quantifies the number of feature differences between Xa and Xb.
      2. The second term measures the degree of agreement between the features of Xa and Xb
         when compared to other data points in the dataset.
      3. The third term normalizes the result by considering the number of differing features
         between Xa and all data points, and similarly for Xb.
    '''
    Xa = X[a,:]
    Xb = X[b,:]

    #first_term = -np.sum(Xa!=Xb)
    first_term_v = -(Xa!=Xb).astype(int)
    first_term   = np.sum(first_term_v)
    mask         = np.ones(len(X), bool)
    mask[[a,b]]  = False
    second_term_v = np.sum(np.sign(Xa-X[mask,:])*np.sign(Xb-X[mask,:]),0)
    second_term  = np.sum(second_term_v) 
    
    third_term_1 = np.sum(Xa!=X) 
    third_term_2 = np.sum(Xb!=X) 
    
    dabj = 1/(2*np.shape(X)[1]) - (first_term_v + second_term_v)/(2*np.sqrt(third_term_1*third_term_2 + np.finfo(float).eps))
    
    dab  = 1/2 - (first_term + second_term)/(2*np.sqrt(third_term_1*third_term_2 + np.finfo(float).eps))

    return dab, dabj

def get_GDM(X):
    '''
    Calculate the Generalized Distance Matrix (GDM) for a given dataset.
    
    Parameters:
    X (numpy.ndarray): A matrix containing NIHSS scores, where in rows there are subjects, 
                        while in columns there are NIHSS items.
    Returns:
    GDM (numpy.ndarray): A symmetric matrix representing the Generalized Distance Matrix.
                        The GDM measures the pairwise distances between data points in X.
    '''

    GDM = np.zeros(shape = ( len(X), len(X)))
    for a in range(len(X)):
        for b in range(a):
            GDM[a,b], _ = dab(a,b,X)

    GDM = GDM + GDM.T
    return GDM

##################
### CLUSTERING ###
##################
# RSC
class sphericalKMeans:
    def __init__(self, n_clusters, max_iter=300, n_init=10, tol=1e-4, verbose=False, init="rand_data", random_state=32412):
        """
        KMeans Class constructor.

        Args:
          n_clusters (int) : Number of clusters used for partitioning.
          iters (int) : Number of iterations until the algorithm stops.

        """
        self.n_clusters       = n_clusters
        self.max_iter         = max_iter
        self.n_init           = n_init
        self.cluster_centers_ = None
        self.labels_          = None
        self.inertia          = 0
        self.iter_conv        = None
        self.tol              = tol
        self.spherinit        = init
        self.seed             = random_state
        self.rng              = np.random.default_rng(self.seed)
        
    def centroids_rand_init(self, X, n_clusters):
        # rng = np.random.default_rng(self.seed)
        if self.spherinit == "rand_data":
            # rand_indices = rng.choice(a=len(X), size=n_clusters, replace=False)
            # centroids = X[rand_indices]
            centroids = self.rng.choice(a=X, size=n_clusters, replace=False, shuffle=False)
            return centroids
        
        elif self.spherinit == "rand_unif":
            norms = np.zeros(n_clusters)
            while 0. in norms:
                centroids = self.rng.standard_normal(size=(n_clusters, X.shape[1]))
                centroids_sq = centroids**2
                norms = np.sqrt(centroids_sq.sum(axis=1))
            return centroids/norms[:,np.newaxis]
        
        elif self.spherinit == "rand_clust":
            rand_X = self.rng.permutation(X)
            clust_dim = len(rand_X) // n_clusters
            clust_dim_last = clust_dim + (len(rand_X) % n_clusters)
            centroids = []
            for n in range(n_clusters-1):
                centroid = np.mean(X[n*clust_dim:(n+1)*clust_dim], axis=0)
                centroids.append(centroid)
            centroids.append(np.mean(X[(n+1)*clust_dim:], axis=0))
            return np.array(centroids)
        
        elif self.spherinit == "k-means++":
            centroids = []
            rand_idx = self.rng.integers(low=0, high=len(X))
            centroids.append(X[rand_idx])
            Y = np.delete(X, rand_idx, axis=0)
            for i in range(n_clusters-1):
                probs_weight = np.ones(len(Y))
                for point_idx in range(len(Y)):
                    best_dist = []
                    for centroid in centroids:
                        dist = np.dot(X[point_idx], centroid)
                        if dist > 1:
                            dist = 1.0
                        if dist < -1:
                            dist = -1.0
                        best_dist.append(dist)
                    probs_weight[point_idx] = np.arccos(np.max(best_dist))**2
                    # print(best_dist)
                    # print(np.max(best_dist))
                # print(probs_weight)
                probs_weight = probs_weight / np.sum(probs_weight)
                # print(probs_weight)
                rand_idx = self.rng.choice([idx for idx in range(len(Y))], p=probs_weight)
                centroids.append(Y[rand_idx])
                Y = np.delete(Y, rand_idx, axis=0)
            return np.array(centroids)

    def find_closest_centroids(self, X, centroids):
        labels = np.zeros(len(X), dtype=int)
        for point_idx in range(len(X)):
            best_sim = []
            for centroid in centroids:
                sim = np.dot(X[point_idx], centroid)
                best_sim.append(sim)
            best_sim = np.array(best_sim)
            labels[point_idx] = np.argmax(best_sim)
            inertia = np.sum(best_sim)
        return labels, inertia

    def compute_centroids(self, X, labels):
        centroids = []
        for label in range(self.n_clusters):
            m = np.sum(X[labels==label], axis=0) / (len(X[labels==label]) + 1e-8)
            c = (m) / (np.linalg.norm(m) + 1e-8)
            centroids.append(c)
        return np.array(centroids)

    def fit(self, X):
        best_of_inits_inertia = -np.inf
        
        for init in range(self.n_init):           
            # Compute initial position of the centroids
            centroids = self.centroids_rand_init(X, self.n_clusters)
            labels = np.zeros(len(X), dtype=int)
            prev_inertia = 0
            prev_centroids = centroids
            for i in range(self.max_iter):
                # For each example in X, assign it to the closest centroid
                labels, inertia = self.find_closest_centroids(X, centroids)
                # Given the memberships, compute new centroids
                centroids = self.compute_centroids(X, labels)
                # Check if centroids stopped changing positions
                if np.linalg.norm((centroids - prev_centroids), ord="fro") <= self.tol:
                    break
                else:
                    prev_inertia = inertia
                    prev_centroids = centroids
            # end of iters
            best_of_iters_inertia = inertia
            best_of_iters_centers = centroids
            best_of_iters_labels = labels
            if best_of_iters_inertia >= best_of_inits_inertia:
                best_of_inits_inertia = best_of_iters_inertia
                best_centers = best_of_iters_centers
                best_labels = best_of_iters_labels
        # end of inits 
        self.inertia          = best_of_inits_inertia
        self.cluster_centers_ = best_centers
        self.labels_          = best_labels
        # self.iter_conv        = iter_conv

        return self
    
class RSC():
    '''
    Repeated Spectral Clustering (RSC) class for spectral clustering of weighted graphs.
    '''

    def __init__(self, k, L_type="sym", spherical=False, N=1000, n_init=4, max_iter=300, seed = 32412, init="k-means++"):
        self.k         = k
        self.N         = N
        self.L_type    = L_type
        self.spherical = spherical
        self.n_init    = n_init
        self.seed      = seed
        self.rng       = np.random.default_rng(self.seed)
        self.max_iter  = max_iter
        self.D         = None
        self.D_C       = None
        self.L_W       = None
        self.T         = None
        self.U         = None
        self.UtotW     = None
        self.UtotC     = None
        self.C         = None
        self.Csum      = None
        self.U_C       = None
        self.T_C       = None
        self.kmeans    = None
        self.gap_C     = None
        self.inertia   = None
        self.eigW      = None
        self.eigC      = None
        self.init      = init
        if self.L_type != "sym":
            self.spherical = False

    def get_laplacian(self, W):
        '''
        Compute the Laplacian matrix of a given weighted graph.

        Parameters:
        W (numpy.ndarray): The weighted adjacency matrix representing the graph.
        L_type (str, optional): The type of Laplacian matrix to compute.
                                    - 'sym' (default): Symmetrically normalized Laplacian.
                                    - 'rw': Random walk Laplacian.
                                    - 'unnorm': Unnormalized Laplacian.
        '''
        D = np.diag(np.sum(W, axis=1))

        if self.L_type == 'rw':
            D_ = np.linalg.inv(D)
            L  = D_ @ (D - W)

        elif self.L_type == 'sym':
            D_     = np.linalg.inv(D)
            D_sqrt = np.sqrt(D_)
            L      = D_sqrt @ (D - W) @ D_sqrt

        elif self.L_type == 'unnorm':
            L = D - W

        return L, D

    def spectral_embedding(self, L):
        '''
        Perform spectral decomposition on a given weighted graph.

        Parameters:
        W (numpy.ndarray): The weighted adjacency matrix representing the graph.
        K (int): The number of eigenvectors to compute for the spectral decomposition.

        Returns:
        T (numpy.ndarray): The spectral embedding matrix containing K eigenvectors.
                            Each row of T represents the spectral embedding of a graph node.
        eig (numpy.ndarray): The K smallest eigenvalues of the Symmetrical Laplacian matrix.
        '''

        # rng = np.random.default_rng(self.seed)
        v0 = self.rng.random(L.shape[0])

        if self.L_type == "sym":
            # eig, eigv = linalg.eigsh(L, k = self.k+1, which = "SM", v0 = v0)
            eigL, eigvL = linalg.eigsh(L, k=len(L)-1, which="SM", v0=v0)
            Utot = np.real(eigvL)
            U = np.real(eigvL[:,:self.k])
            U2 = U**2
            row_sums = np.sqrt(U2.sum(axis=1))
            T = U/row_sums[:, np.newaxis]
        else:
            # eig, eigv = linalg.eigsh(L, k = self.k+2, which = "SM", v0 = v0)
            eigL, eigvL = linalg.eigsh(L, k=len(L)-1, which="SM", v0=v0)
            U = np.real(eigvL[:,:self.k+1])
            Utot = np.real(eigvL)
            T = U[:,1:]
        return U, T, eigL, Utot

    def repeated_kmeans(self, T):
        '''
        Perform repeated K-Means clustering on a given dataset.

        Parameters:
        T (numpy.ndarray): The input data matrix for clustering.
        N (int): The number of repetitions for K-Means clustering.
        k (int): The number of clusters (centroids) to fit in each K-Means clustering.
        n_init (int, optional): The number of times K-Means will be run with different centroid seeds.
                                Defaults to 10.
        seed (int, optional): Seed value for random initialization of K-Means centroids.
                                Defaults to 32412.

        Returns:
        C (numpy.ndarray): A 3D array containing the co-occurrence counts of data points belonging to the same cluster.
                        C[i, j, l] represents the co-occurrence count of data points j and l in repetition i.
        '''

        # rng = np.random.default_rng(self.seed)
        C = np.zeros((self.N,T.shape[0],T.shape[0]))
        list_inertia = np.zeros(self.N)
        
        for i in range(self.N):
            if self.spherical:
                kmeans = sphericalKMeans(n_clusters=self.k, init=self.init, n_init=self.n_init, max_iter=self.max_iter, random_state=self.rng.integers(low=0, high=1e8)).fit(T)
            else:
                kmeans = cluster.KMeans(n_clusters=self.k, init=self.init, n_init=self.n_init, max_iter=self.max_iter, random_state=self.rng.integers(low=0, high=1e8), verbose=0).fit(T)

            list_inertia[i] = float(kmeans.inertia_)
            # matrix of the clustered subjects, cluster label is one-hot-encoded
            K = np.zeros((len(kmeans.labels_), len(kmeans.cluster_centers_)))
            for label in range(K.shape[1]):
                K[kmeans.labels_==label,label] = 1
            
            # co_occurrence of labels is np.dot(pres_matrix.T, pres_matrix)
            C[i] = K @ K.T
            np.fill_diagonal(C[i], 0)
        return C, list_inertia

    def fit(self, W):
        '''
            RSC pipeline.
        '''
        self.L_W, self.D                          = self.get_laplacian(W)
        self.U, self.T, self.eigW, self.UtotW     = self.spectral_embedding(self.L_W)
        self.C, self.inertia                      = self.repeated_kmeans(self.T)
        self.Csum                                 = np.sum(self.C,axis=0)
        L_C, self.D_C                             = self.get_laplacian(self.Csum)
        self.U_C, self.T_C, self.eigC, self.UtotC = self.spectral_embedding(L_C)
        self.gap_C                                = self.eigC[self.k] - self.eigC[self.k-1]
        
        if self.spherical:
            self.kmeans = sphericalKMeans(n_clusters=self.k, n_init=self.n_init, init=self.init, max_iter=self.max_iter, random_state=self.seed).fit(self.T_C)
        else:
            self.kmeans  = cluster.KMeans(n_clusters=self.k, n_init=self.n_init, init=self.init, max_iter=self.max_iter, random_state=self.seed).fit(self.T_C)
        return self

#####################
### VISUALIZATION ###
#####################

# CORRELATIONS NETWORK #
def correlation_net(corrdata, spec_dict):
    '''
    This function creates a network plot with the co-occurrences (edges) between item deficits (nodes).
    It takes as input a dataframe with named indexes and columns, and the title for the plot.
    It returns a figure object to be saved.

    Parameters:
    corrdata (pandas.DataFrame): table of pairwise correlations between NIHSS items, indexed on axes 0 and 1.
    spec_dict (dict): A dictionary with plot parameters.

    Returns:
    fig, ax (matplotlib figure, Axes).
    '''
    cmap = matplotlib.cm.coolwarm  # Colormap to use
    reorder = [6, 5, 11, 1, 0, 7, 8, 3, 2, 13, 4, 12, 9, 10]
    nihss_columns = list(corrdata.columns.values)
    corrdataf = corrdata.loc[[nihss_columns[m] for m in reorder], [nihss_columns[m] for m in reorder]].copy()

    # Prepare graph edges from correlation matrix
    gve_df = [{"V1": corrdataf.index[i], 
               "V2": corrdataf.columns[j], 
               "Edgestrength": corrdataf.iloc[i, j], 
               "Correlation coeff": corrdataf.iloc[i, j]}
               for i in range(len(corrdataf)) for j in range(i)]
    gve_df = pd.DataFrame(gve_df)

    # Create the graph
    fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]),
                           constrained_layout=True)
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])

    G = nx.from_pandas_edgelist(gve_df, source='V1', target='V2', edge_attr='Edgestrength')

    # Edge labels: Only show non-zero edges
    labels = {edge: np.round(corrdataf.at[edge[0], edge[1]], 2)
              for edge in G.edges if G[edge[0]][edge[1]]['Edgestrength'] != 0} if spec_dict['labeled'] else {}

    # Position nodes in a circular layout
    theta = np.linspace(start=2*np.pi/(2*len(corrdataf.columns)), 
                        stop=2*np.pi + 2*np.pi/(2*len(corrdataf.columns)), 
                        num=len(corrdataf.columns), endpoint=False)
    pos = {node: [np.cos(angle), np.sin(angle)] for node, angle in zip(corrdataf.columns, theta)}

    # Edge widths: Based on strength of correlation
    edge_widths = [spec_dict['widthmultiplier']*np.abs(G[u][v]['Edgestrength']) for u, v in G.edges]

    # Adjust node positions for labels
    pos_higher = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}

    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    # Map edge strengths to colors using the colormap
    edge_colors = [G[u][v]['Edgestrength'] for u, v in G.edges]
    edge_colors = [cmap(norm(edge)) for edge in edge_colors]
    
    # Draw the graph
    nx.draw_networkx_labels(G, pos_higher, font_size=spec_dict['font'] - 4)
    nx.draw(G, pos, edge_color=edge_colors, width=edge_widths, with_labels=False,
            node_color=spec_dict['nodecolor'], node_size=spec_dict['nodesize'], 
            edge_cmap=cmap)

    # Draw edge labels if applicable
    if spec_dict['labeled']:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=spec_dict['font'] - 6)

    # Add colorbar
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])  # Needed for ScalarMappable
    cbar = plt.colorbar(mapper, orientation="vertical", ax=ax, shrink=spec_dict['shrink'])
    cbar.set_label("Spearman correlation", fontsize=spec_dict['font'] - 2)
    cbar.set_ticks(np.arange(-1, 1+spec_dict['cbarticks'], spec_dict['cbarticks']))
    cbar.ax.tick_params(axis='both',labelsize=spec_dict['font']-6)

    # Hide axis
    ax.set_axis_off()

    return fig, ax

# PATIENTS NETWORKS #
def subject_network_plot(adjacency_matrix_untresh, spec_dict,
                         node_names=None, labels=None):
    '''
    This function plots a network of subjects connected by according to an adjacency matrix, eventual clustering labels.

    Parameters:
    adjacency_matrix (2-D array-like): the adjacency matrix describing the network to plot.
    N (int): the number of repetitions for co-occurrences graphs. Defaults to 1.
    node_names (list of strings): the names of nodes in the network, their subject-specific labels.
    node_color_labels (list of levels): the codes according to which colors are assigned, clustering labels to have corresponding nodes colors.
    edge_cmap (string): name of a colormap in matplotlib, or "warm_cmap" to use the upper part of "cool_warm".
    title (string): title of the figure.
    cbar_title (string): title of the colorbar, on the side.
    thresh (int): the threshold to cut-off co-occurrences, if any. Defaults to 0.
    F1 (float): a coefficient to scale edge_width according to edge weights (adjacency entries).
    F2 (float): a coefficient to scale edge_color from the colorbar, according to edge weights (adjacency entries).
    iters (int): iterations for the Fruchtermann-Reingold algorithm.
    kdist (float): coefficient k for the nodes distances.
    cluster_legend (list of strings): allows for other color labels.

    Returns:
    fig: a matplotlib figure.
    ax: a matplotlib Axes.
    '''

    adjacency_matrix = adjacency_matrix_untresh.copy()
    adjacency_matrix[adjacency_matrix<spec_dict['thresh']] = 0
    
    ourcmap = ListedColormap(plt.cm.coolwarm(np.linspace(0.5+spec_dict['thresh']/spec_dict['N'],1,100)))
        
    adjacency_in_use = adjacency_matrix.copy()
    np.fill_diagonal(adjacency_in_use, 0)
    nonzero_weights = adjacency_in_use[adjacency_in_use!=0]
    
    # create graph
    fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0],spec_dict['figdim'][1]))
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'], fontweight=spec_dict['fontweight'])
    
    G2      = nx.from_numpy_array(adjacency_in_use, create_using=nx.Graph)
    pos2    = nx.layout.spring_layout(G2, seed=spec_dict['seed'], iterations=spec_dict['iters'], k=spec_dict['kdist'])
    labels2 = nx.get_edge_attributes(G2, "weight")

    widths2 = list(np.array(list(labels2.values()))*spec_dict['widthmultiplier']/spec_dict['N'])
    colors2 = list(np.array(list(labels2.values()))/spec_dict['N'])  
    
    # color edges    
    if node_names is not None:
        node_names = {n:lab for n,lab in zip(G2,node_names)}
    else:
        node_names = {n:lab for n,lab in zip(G2,[i for i in range(len(adjacency_in_use))])}
    nx.draw_networkx_labels(G2, pos2, font_size=spec_dict['font']-10, labels=node_names, ax=ax)
    
    if labels is not None:
        if spec_dict['K']<=10:
            colori = [plt.cm.tab10(spec_dict['cluster_colors'][i]) for i in labels]
        else:
            colori = [plt.cm.tab20(spec_dict['cluster_colors'][i]) for i in labels]
            
        nx.draw_networkx_nodes(G2, pos2, node_color=colori, 
                               node_size=spec_dict['nodesize'], alpha=spec_dict['alpha'], ax=ax)
    else:
        nx.draw_networkx_nodes(G2, pos2, node_color='lightblue', 
                               node_size=spec_dict['nodesize'], alpha=spec_dict['alpha'], ax=ax)
        
    nx.draw_networkx_edges(G2, pos2, edgelist=labels2.keys(), width=widths2, 
                           edge_color=colors2, edge_cmap=ourcmap, edge_vmin=0, edge_vmax=1, ax=ax)

    # COLORBAR EDGES
    sm = plt.cm.ScalarMappable(cmap=ourcmap, norm=plt.Normalize(vmin = spec_dict['thresh'], vmax=spec_dict['N']))
    #clb = plt.colorbar(sm, location="right", shrink=0.9, pad=0.02, anchor=(0.,1.), ax=plt.gca())
    clb = plt.colorbar(sm, location="right", anchor=(0.,1.), ax=plt.gca())
    clb.set_ticks(np.arange(0,spec_dict['N']+spec_dict['cbarticks'],spec_dict['cbarticks']))
    clb.set_ticklabels(np.arange(0,spec_dict['N']+spec_dict['cbarticks'],spec_dict['cbarticks']),fontsize=spec_dict['font']-4)
    clb.ax.set_ylabel(spec_dict['cbar_title'],fontsize=spec_dict['font']-2, fontweight=spec_dict['fontweight'])

    # COLORBAR CLUSTERS
    if spec_dict['cluster_colorbar']:            
        if spec_dict['cluster_legend'] is None:
            cluster_legend = []*spec_dict['K']
            for i in range(spec_dict['K']):
                cluster_legend.append("C"+str(i))
            spec_dict['cluster_legend'] = cluster_legend
               
        unique_colori = [plt.cm.tab10(i) for i in spec_dict['cluster_colors']]

        sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(unique_colori), norm=plt.Normalize(vmin=0, vmax=len(unique_colori)))
        clb2 = plt.colorbar(sm,
                           location="bottom",
                           fraction=0.08,
                           shrink=0.8,
                           pad=0.02,
                           ax=plt.gca())
        clb2.set_ticks(np.arange(0.5,len(unique_colori),1))
        clb2.set_ticklabels(spec_dict['cluster_legend'], fontsize=spec_dict['font']-6, fontweight=spec_dict['fontweight'])
        clb2.ax.set_xlabel(spec_dict['cbarcluster_title'], fontsize=spec_dict['font']-2, fontweight=spec_dict['fontweight'])
            
    return fig, ax

# RADARPLOT #
def cluster_radarplotter(fig_polar, ax, cluster_df, spec_dict):
    '''
    Create a radar plot (spider plot) to visualize cluster characteristics.

    Parameters:
    fig_polar (matplotlib.figure.Figure): The polar figure to which the radar plot will be added.
    cluster_df (pandas.DataFrame): The DataFrame containing cluster data for plotting.
    title (str): The title of the radar plot.
    minmaxscale (bool, optional): Whether to scale the data to [0, 1] (True) or use original values (False).
                                  Defaults to True.

    Returns:
    None

    '''

    feat_ordered =  ['Motor Leg R', 'Motor Arm R','Best Language','LOC-C','LOC-Q',
                     'Motor Arm L', 'Motor Leg L','Visual','Best Gaze','Inattention',
                     'Facial Palsy','Dysarthria', 'Limb Ataxia', 'Sensory']
    n = len(feat_ordered)

    if spec_dict['minmaxscale']:
        maxs       = [4,4,3,2,2,4,4,3,2,2,3,2,2,2]
        r          = 1
        ticks      = np.arange(-0.25,1.25,0.25)
        ticks_lbls = ['','0%','25%','50%','75%','100%']
    else:
        maxs       = 1
        r          = 4
        ticks      = np.arange(-1,r+2)
        ticks_lbls = ['',0,1,2,3,4,'']

    # Plot Grids
    theta14       = np.linspace(start=np.pi/14, stop=np.pi*(2+1/14), num=n, endpoint=False)
    lines, labels = plt.thetagrids(theta14*180/np.pi, labels=feat_ordered, fontsize=spec_dict['font']-4, fontweight=spec_dict['fontweight'])

    # Plot and color Bars
    radii = r*np.ones(n)
    width = (2*np.pi/n)*np.ones(n)
    bars  = plt.bar(theta14, radii, width=width)
    cols  = ['tab:olive']*2+['tab:blue']*3+['tab:orange']*2+['whitesmoke']*2+['tab:pink']*3+['whitesmoke']*2
    for bar, col in zip( bars, cols):
        bar.set_facecolor(col)
        bar.set_alpha(spec_dict['alpha'])

    theta15 = np.append(theta14, theta14[0])
    if isinstance(cluster_df,pd.DataFrame):
        # Calculate statistics
        max_patient = cluster_df.loc[:,feat_ordered].max()/maxs
        min_patient = cluster_df.loc[:,feat_ordered].min()/maxs
        median      = cluster_df.loc[:,feat_ordered].median()/maxs
    
        max_patient = np.append(max_patient, max_patient.iloc[0])
        min_patient = np.append(min_patient, min_patient.iloc[0])
        median      = np.append(median, median.iloc[0])

        # Plot statistics
        m  = plt.plot(theta15, median,      marker='o', markersize=spec_dict['s'], linestyle='-', linewidth=spec_dict['linew'], color='green')
        mx = plt.plot(theta15, max_patient, marker='o', markersize=spec_dict['s'], linestyle=':', linewidth=spec_dict['linew'], color='red')
        mn = plt.plot(theta15, min_patient, marker='o', markersize=0,              linestyle=':', linewidth=spec_dict['linew'], color='blue')
        
        ax.fill_between(theta15, min_patient, max_patient, alpha=spec_dict['alpha'], color='gray')
        ax.fill_between(theta15, min_patient, median,      alpha=spec_dict['alpha'], color='gray')
        ax.legend(
            ['Median Score', 'Max Score', 'Min Score'], 
            bbox_to_anchor=spec_dict['bbox_to_anchor'], 
            prop={'weight': spec_dict['fontweight'], 'size': spec_dict['font']-8}, 
            loc=spec_dict['loc']
        )
        
    elif isinstance(cluster_df,pd.Series):
        # Plot patient profile
        patient_profile = cluster_df.loc[feat_ordered]/maxs
        patient_profile = np.append(patient_profile, patient_profile.iloc[0])
        m  = plt.plot(theta15, patient_profile, marker='_', markersize=spec_dict['alpha'],   linestyle='-', linewidth=spec_dict['linew'], color='black')
        
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks_lbls, fontsize=spec_dict['font']-8, fontweight=spec_dict['fontweight'])
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'], fontweight=spec_dict['fontweight'])
    return fig_polar, ax