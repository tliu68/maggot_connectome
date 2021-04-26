#%%
cd '/Users/TingshanLiu/Desktop/2020 summer/TL_maggot/maggot_connectome/'
#%%
from pkg.data import load_maggot_graph, load_data
from joblib import Parallel, delayed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import networkx as nx
from graspologic.utils import cartesian_product, import_graph, binarize, remove_loops
from graspologic.models import SBMEstimator
from graspologic.plot import heatmap
from graspologic.simulations import er_nm
from graspologic.models.sbm import _get_block_indices, _calculate_block_p, _block_to_full
from sklearn.utils import check_X_y
from scipy.stats import norm, binom_test
from scipy import stats
from hyppo.independence import Dcorr
from hyppo.ksample import KSample
import pandas as pd
from sklearn.preprocessing import normalize
from statistics import mode
# import pymaid
# from pkg.pymaid import start_instance
# %%
context = sns.plotting_context(context="poster", font_scale=1.2)
sns.set_context(context)

# %%


def model_sbm(adj, cls_name):
    if len(cls_name) == 1:
        sbm = SBMEstimator()
        sbm.fit(binarize(adj), y=meta[cls_name[0]].values)
    elif len(cls_name) == 2:  # crossing 'hemisphere' w/ another class
        multi_labels = meta[cls_name[0]].astype(str) + meta[cls_name[1]].astype(str)
        # assuming the first class is 'hemisphere'
        sbm = SBMEstimator()
        sbm.fit(binarize(adj), y=multi_labels.values)
    return sbm


def fdr(p, rate):
    sort_idx = np.argsort(p)
    n = len(p)
    q = np.zeros(n)
    for i in range(n):
        q[i] = sort_idx[i] * rate / n
    threshold = np.where(np.sort(p) < np.sort(q))[0][-1]
    sig_p = np.array(p)[sort_idx][threshold]
    return sig_p


def plot_connectivity(coln, cls, block_pairs, B, p_val, pos, Bs, factor=1.5, correc='B'):
    counts = []
    for i in np.unique(meta[coln]):
        n = np.sum(meta[coln] == i)
        counts.append(n)
    nodes = range(len(np.unique(meta[coln])))
    labels = {}
    new_keys = list(cls.keys())
    nn = int(len(np.unique(meta[coln])) / 2)
    for n in nodes:
        if n < nn:
            labels[n] = ''
        else:
            labels[n] = new_keys[n-nn].split('_')[-1]
    # if correc == 'fdr':
    #     sig_p = fdr(p_val, 0.2)
    # else:
    #     sig_p = 0.05 / len(p_val)
    # sig_p = 0.000005  # top 90
    sig_p = np.min(p_val)
    edges = block_pairs[np.where(np.array(p_val) <= sig_p)[0]]
    edges = [tuple(edges[i,:]) for i in range(edges.shape[0])]
    weights = B.ravel()[np.where(np.array(p_val) <= sig_p)[0]]
    # weights = [((i-np.min(weights)) / (np.max(weights)-np.min(weights))) * factor +0.2
    #     for i in weights]
    weights = (np.log(weights) - np.min(np.log(weights))) / 8
    # cutoff = np.sort(weights)[::-1][130]  # plot heightest 1XX weights
    # weights[weights < cutoff] = 0

    # assign colors based on dominance of edge types
    es = ['aa','ad','da','dd']
    c = ['orange','b','r','g']
    # c = [mpl.colors.to_rgba(i) for i in c]
    # c = [mpl.cm.get_cmap(cmap)(i) for i in [3,0,1,6]]  #0,1,5,4(Set1)
    colors_e = []
    for i in range(len(edges)):
        j = []
        for ii in range(4):
            j.append(Bs[ii][edges[i][0], edges[i][1]])
        colors_e.append(c[np.argmax(j)])
    colors_n = []
    for i in labels:
        colors_n.append(mode(meta[meta[coln]==i]['color']))    

    fig, ax = plt.subplots(1, figsize=(8,24))
    # ax.set_xlim([-28,22])
    # ax.set_ylim([-24,16.5])
    ax.set_aspect('equal', adjustable='box')
    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.draw_networkx_nodes(g, node_size=counts, pos=pos, node_color=colors_n, ax=ax)
    label_pos = pos.copy()
    # offset_size = 1
    # np.random.seed(333)
    offset_sizes = [0.9, 1.4]
    ii = -1
    for i in pos.keys():
        ii += 1
        # for j in range(2):
        #     label_pos[i][j] = pos[i][j] + counts[i]/1200 + offset_size[j]
        # label_pos[i] = [j+counts[i]/1000+offset_size for j in pos[i]]
        label_pos[i] = [pos[i][0], pos[i][1]+counts[i]/1500+ offset_sizes[ii%2]]
    label_pos[42][1] = pos[42][1]-counts[42]/1500 - offset_sizes[0]
    # label_pos[16][1] = label_pos[16][1] + offset_size
    i = -1
    for e in g.edges:
        i += 1
        ax.annotate("",
                    xy=pos[e[1]], xycoords='data',
                    xytext=pos[e[0]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=colors_e[i],
                                    shrinkA=5+counts[e[0]]/50,
                                    shrinkB=5+counts[e[1]]/50,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=0.1",
                                    mutation_scale=10, lw=weights[i],
                                    ),
                    )
    ax.axis('off')
    for i in range(len(edges)):
        if edges[i][0] == edges[i][1]:
            gap = counts[edges[i][0]] / 600 + 0.3
            center = tuple([pos[edges[i][0]][0], pos[edges[i][0]][1]+gap])
            l = mpl.patches.Arc(center,0.3,0.5,0,-60,240,
                lw=weights[i], color=colors_e[i])
            ax.add_patch(l)
    nx.draw_networkx_labels(
        g, pos=label_pos, labels=labels, font_color='y', ax=ax, horizontalalignment='left', font_size=10
    )
    ypos_ = [11,7,0.5,-8,-19,-23]
    texts = ['Input', '2nd-Order', '3rd-Order','4th-Order','Nth-1 Order','Nth Order']
    for i in range(len(texts)):
        ax.text(-26, ypos_[i], texts[i], ha='center', va='center', fontsize=13)
    x = ax.get_xlim()
    y = ax.get_ylim()
    ax.set_xlim([ax.get_xlim()[0]-np.ptp(x)/20, ax.get_xlim()[1]+np.ptp(x)/20])
    ax.set_ylim([ax.get_ylim()[0]-np.ptp(y)/50, ax.get_ylim()[1]+np.ptp(y)/50])
    ax.text(x[0] - np.ptp(x)/100, y[1], 'L', fontsize=13)
    ax.text(x[1] + np.ptp(x)/100, y[1], 'R', fontsize=13)
    for i in range(4):
        ax.text(x[1] - np.ptp(x)/20, y[0] + np.ptp(y)/40*i, es[i], c=c[i], fontsize=12)
    ax.text(x[1] - np.ptp(x)/8, y[0] + np.ptp(y)/40*4, 'dominant edge type:', fontsize=12)
    # ax.text(15, -23+4, 'dominant edge type:', fontsize=13, ha='center')
    # for i in range(4):
    #     ax.text(15, -23+i, es[i]+': '+str(ns[i]), color=c[i], fontsize=13, ha='left')
    # ax.text(15,-24, 'tie'+': '+str(ns[-1]), c='k', ha='left', fontsize=13)
    # plt.show()
    plt.savefig('connectivity-hemi_norm', transparent=False, facecolor='white', bbox_inches = "tight", dpi=300)

def _calculate_block_p_wei(graph, block_inds, block_vert_inds, return_counts=False):
    n_blocks = len(block_inds)
    block_pairs = cartesian_product(block_inds, block_inds)
    block_p = np.zeros((n_blocks, n_blocks))
    for p in block_pairs:
        from_block = p[0]
        to_block = p[1]
        from_inds = block_vert_inds[from_block]
        to_inds = block_vert_inds[to_block]
        block = graph[from_inds, :][:, to_inds]
        p = np.sum(block) / block.size
        block_p[from_block, to_block] = p
    return block_p

def weighted_sbm(adj, y):
    graph = import_graph(adj)
    check_X_y(graph, y)
    graph = remove_loops(graph)
    block_vert_inds, block_inds, block_inv = _get_block_indices(y)
    block_p = _calculate_block_p_wei(graph, block_inds, block_vert_inds)
    p_mat = _block_to_full(block_p, block_inv, graph.shape)
    p_mat = remove_loops(p_mat)
    return block_p, p_mat

def generate_rand_block_p_wei(adj, y, rep=100, n_jobs=5):
    def _run():
        g = er_nm(len(adj), np.count_nonzero(adj), directed=True)
        edges = adj[np.nonzero(adj)]
        np.random.shuffle(edges)
        g[np.nonzero(g)] = edges
        r_block_p, _ = weighted_sbm(g, y)
        return r_block_p
    rand_block_p = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_run)()
        for _ in range(rep))
    return rand_block_p

def compute_block_sig_wei(block_p, adj, y, n_rand=100):
    _, block_inds, _ = _get_block_indices(y)
    rand_block_p = generate_rand_block_p_wei(adj, y, n_rand)
    block_pairs = cartesian_product(block_inds, block_inds)
    p_val = []
    for p in block_pairs:
        null = np.array([i[p[0], p[1]] for i in rand_block_p])
        if np.std(null) == 0:
            p_val.append(1)
        else:
            z_score = (block_p[p[0], p[1]] - np.mean(null)) / np.std(null)
            p_val.append(1 - norm.cdf(z_score))
        # p_val.append((np.sum(null >= block_p[p[0], p[1]]) + 1) / (n_rand + 1))
    return p_val, block_pairs


#%%[markdown]
# ### load the data
# %%
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"]]
mg.fix_pairs()
meta = mg.nodes.copy()
adj = mg.sum.adj.copy()

#%%[markdown]
# ### grap information on labels from different columns of meta
# %%
class_4 = {'input_AN': [["['sens_subclass_AN']"], 'all_class2'],
    'input_olf': [["['sens_subclass_ORN']"], 'all_class2'],
    'input_MN': [["['sens_subclass_MN']"], 'all_class2'],
    'input_thermo': [["['sens_subclass_thermo']"], 'all_class2'],
    'input_photo': [["['sens_subclass_photoRh5']",
        "['sens_subclass_photoRh6']"], 'all_class2'],
    'input_vtd': [["['sens_subclass_vtd']"], 'all_class2'],
    'input_ascending': [['ascendings'], 'simple_group'],
    'interneuron_uPN': [['uPN'], 'merge_class'],
    'interneuron_ALN': [['bLN','pLN','cLN'], 'merge_class'],
    'interneuron_unkLN': [],
    'interneuron_mPN': [['mPN-multi', 'mPN-olfac'], 'merge_class'],
    'interneuron_tPN': [['tPN'], 'merge_class'],
    'interneuron_vPN': [['vPN'], 'merge_class'],
    'interneuron_LON': [['LON'], 'merge_class'],
    'interneuron_unkPN': [],
    'interneuron_LHN': [['LHNs'], 'simple_group'],
    'interneuron_KC': [['KCs'], 'simple_group'],
    'interneuron_MBIN': [['MBINs'], 'simple_group'],
    'interneuron_MBON': [['MBONs'], 'simple_group'],
    'interneuron_FBN': [['MB-FBNs'], 'simple_group'],
    'interneuron_CN': [['CNs'], 'simple_group'],
    'interneuron_unk': [['unk'], 'simple_group'],
    'output_pre-dVNC': [['pre-dVNCs'], 'simple_group'],
    'output_pre-dSEZ': [['pre-dSEZs'], 'simple_group'],
    'output_dVNC': [['dVNCs'], 'simple_group'],
    'output_dSEZ': [['dSEZs'], 'simple_group'],
    'output_RGN': [['RGNs'], 'simple_group'],
}

#%%
all_classes = {'class_4': class_4}
for ii, kk in enumerate(all_classes.keys()):
    meta[kk] = ''
    for idx, (k, v) in enumerate(all_classes[kk].items()):
        if len(v) == 1:
            meta.loc[meta[v[0]].eq(True).any(1), kk] = idx
        elif len(v) > 1:
            for i in range(len(v[0])):
                meta.loc[meta[v[1]].eq(v[0][i]), kk] = idx

# Some nodes that labeled as "PN" or "LN" in meta['simple_group']
# are labeled as things like "AN2", "MN2" etc in meta['merge_class']
# that do not specify which "PN" or "LN"
# even though meta['merge_class'] gives detailed labels for the majority of nodes
# so in order not to exclude any node, we will label them as "unkPN" or "unkLN"
idx = list(class_4.keys()).index("interneuron_unkPN")
meta.loc[(meta['class_4'].eq('') & meta['simple_group'].eq('PNs')), 'class_4'] = idx
idx = list(class_4.keys()).index("interneuron_unkLN")
meta.loc[(meta['class_4'].eq('') & meta['simple_group'].eq('LNs')), 'class_4'] = idx
print(np.sum(meta['class_4'].ne('')), len(meta))

#%%
meta['hemi-class_4'] = ''
# meta['hemi-class_4'] = meta['hemisphere'].astype(str) + meta['class_4'].astype(str)
meta.loc[meta['hemisphere']=='L', 'hemi-class_4'] = meta['class_4']
n = len(np.unique(meta['class_4']))
meta.loc[meta['hemisphere']=='R', 'hemi-class_4'] = meta['class_4'] + n

#%%[markdown]
# ### fit an SBM for each category and test significance on the B_ij's

#%%
# # normalize by sum of axon_input & dendrite_input to get the sum graph
# adj_norm = np.zeros(adj.shape)
# for i in range(len(adj)):
#     a_input = meta.iloc[i]['axon_input']
#     d_input = meta.iloc[i]['dendrite_input']
#     if a_input + d_input != 0:
#         adj_norm[:,i] = adj[:,i] / (a_input + d_input)

#%%
# normalize each graph type separately by axon/dendrite input
edge_types = ['aa','ad','da','dd']
adjs = []
for e in edge_types:
    adjs.append(mg.to_edge_type_graph(e).adj)

adjs_norm = []
for i in range(4):
    adjs_norm.append(np.zeros(adj.shape))

for i in range(len(adj)):
    a_input = meta.iloc[i]['axon_input']
    d_input = meta.iloc[i]['dendrite_input']
    if a_input != 0:
        adjs_norm[0][:,i] = adjs[0][:,i] / a_input
        adjs_norm[2][:,i] = adjs[2][:,i] / a_input
    
    if d_input != 0:
        adjs_norm[1][:,i] = adjs[1][:,i] / d_input
        adjs_norm[3][:,i] = adjs[3][:,i] / d_input

adjs_sum_norm = np.sum(adjs_norm, axis=0)
adjs_sum_norm = normalize(adjs_sum_norm, axis=0, norm='l1')


#%%[markdown]
# #### try to incorporate the edge weights
# in the weighted version, each $B_ij =$ sum of edge weights / no. possible edges
# for significance test, currently using one-sided z-test
#%%
input_adj = adjs_sum_norm.copy()
block_p, p_mat = weighted_sbm(
    input_adj, y=meta['hemi-class_4'].values
)

#%%
p_val_bootstrap, block_pairs = compute_block_sig_wei(
    block_p, input_adj, meta['hemi-class_4'].values, 500
)

#%%
fig,ax = plt.subplots(1, figsize=(13,13))
heatmap(block_p, ax=ax, norm=LogNorm())
m = np.min(p_val_bootstrap)
for i in range(len(block_pairs)):
    if p_val_bootstrap[i] == m:
        ax.text(block_pairs[i,1], block_pairs[i,0]+0.7,'*', color='w', fontsize=10)
        
#%%
cls = class_4
cls_name = 'class_4'
labels = list(cls.keys())
new_labels = [i.split('_')[-1] for i in labels]
split = [7, 24]
# plot_heatmap(block_p, new_labels, p_val_bootstrap, block_pairs)

#%%
pos = {0:[0,11], 1:[2.5,11], 2:[5,11], 3:[7.5,11], 4:[10,11], 5:[12.5,11], 6:[15,11],
    7:[0,7], 8:[2.5,7], 9:[5,7], 10:[7.5,7], 11:[10,7], 12:[12.5,7], 13:[15,7], 14:[17.5,7],
    15:[0,1], 16:[4,1], 17:[8,3], 18:[8,-1.5], 19:[12,1],
    20:[4.5,-8], 21:[12,-13],
    22:[1,-19],
    23:[4.5,-19], 
    24:[1,-23], 25:[4.5,-23], 26:[8,-23]}

#%%
coord = np.array([i[1] for i in list(pos.items())])
gap = 3
coord = np.vstack((np.vstack((-coord[:,0]-gap, coord[:,1])).T, coord))
pos_hemi = {}
y = np.unique(meta['hemi-class_4'])
# ids = np.array([int(i[1:]) for i in y])
for i in range(len(y)):
    pos_hemi[i] = coord[i]


#%%
# color the edges based on edge type
edge_types = ['aa','ad','da','dd']
# adjs = []
Bs = []
for e in edge_types:
    adj_ = mg.to_edge_type_graph(e).adj
    B_, _ = weighted_sbm(adj_, y=meta['hemi-class_4'].values)
    Bs.append(B_)

#%%
# B_vec = np.vstack(([Bs[i].ravel() for i in range(len(Bs))])).T
# B_max = np.argmax(B_vec, axis=1)
# idx = np.zeros(len(B_max))
# for ii in range(len(B_vec)):
#     if np.all(B_vec[ii,0] == B_vec[ii,:]):
#         idx[ii] = 1
# ns = [np.sum(B_max == i) for i in range(4)]
# ns[0] = int(ns[0] - np.sum(idx))
# # number of blocks where one edge type dominates (:4) or there is a tie (4:5)
# ns.append(len(B_max) - np.sum(ns))

#%%
plot_connectivity('hemi-class_4', class_4, block_pairs, block_p, p_val_bootstrap, pos_hemi, Bs)


#%%
connector_path = "/Users/TingshanLiu/Desktop/2020 summer/TL_maggot/data/2021-03-10/connectors.csv"
connectors = pd.read_csv(connector_path)
connectors = connectors.fillna(-1)
connectors["postsynaptic_to"] = connectors["postsynaptic_to"].astype('int')
connectors["presynaptic_to"] = connectors["presynaptic_to"].astype('int')

#%%
avg_locs = np.zeros((len(meta), 3))
for i,id_ in enumerate(meta.index):
    coords = connectors[
        (connectors["presynaptic_to"] == id_) & (connectors["presynaptic_to"] == id_)
    ][['x', 'y', 'z']].values
    if np.sum(coords) > 0:  # make sure we have coord info for this neuron
        avg_coord = np.mean(coords, axis=0)
        avg_locs[i,:] = avg_coord

print('no coord info for ', np.sum(np.min(avg_locs, axis=1) == 0), ' neurons')

# %%
cls = 'hemi-class_4'
labels = np.unique(meta[cls])
region_avg_locs = np.zeros((n, 3))
for i,label in enumerate(labels):
    idx = np.where([meta[cls] == label])[1]
    coords = avg_locs[idx, :]
    region_avg_locs[i,:] = np.mean(coords[coords.any(axis=1)], axis=0)



#%%
with plt.style.context(('ggplot')):
    fig = plt.figure(figsize=(10,7))
    ax = Axes3D(fig)
    for i in range(len(region_avg_locs)):
        xi = region_avg_locs[i,0]
        yi = region_avg_locs[i,1]
        zi = region_avg_locs[i,2]
        ax.scatter(xi, yi, zi, s=counts[i], c=colors[i])
        # ax.scatter(xi, yi, zi, c=colors[key], s=20+20*G.degree(key), edgecolors='k', alpha=0.7)


# %%
fig,ax = plt.subplots(1, figsize=(28,28))
ax.set_aspect('equal', adjustable='box')
g = nx.MultiDiGraph()
g.add_nodes_from(range(len(region_avg_locs)))
p_k = {}
for i in range(len(labels)):
    p_k[i] = region_avg_locs[i,:2]
colors_n = []
for i in labels:
    colors_n.append(mode(meta[meta['hemi-class_4']==i]['color']))    
counts = []
for i in labels:
    n = np.sum(meta[cls] == i) * 3
    counts.append(n)
nx.draw_networkx_nodes(g, pos=p_k, node_size=counts, node_color=colors_n, ax=ax)
label_pos = p_k.copy()
offset_size = 1000
for i in p_k.keys():
    label_pos[i] = [j+counts[i]*0.1+offset_size for j in p_k[i]]
# label_pos[15][1] = label_pos[15][1] + offset_size
# label_pos[16][1] = label_pos[16][1] + offset_size
node_labels = []
n = int((len(labels)-1) / 2)
l = np.unique(meta['class_4'])
for i in range(n+1):
    name = list(class_4.keys())[int(labels[i][1:])].split('_')[-1]
    node_labels.append(name)
for i in range(n+1, len(labels)):
    node_labels.append('')
node_labels_k = {}
for i in range(len(node_labels)):
    node_labels_k[i] = node_labels[i]
node_labels_k[0] = node_labels_k[0] + ' (C)'
node_labels_k[24] = node_labels_k[24] + ' (L)'
nx.draw_networkx_labels(g, pos=label_pos, labels=node_labels_k, font_color='y', ax=ax, font_size=20)
idx = np.where(np.array(p_val_bootstrap) == np.min(p_val_bootstrap))[0]
edges = block_pairs[idx]
edges = [tuple(edges[i,:]) for i in range(edges.shape[0])]
g.add_edges_from(edges)
weights = block_p.ravel()[idx]
weights = (np.log(weights) - np.min(np.log(weights))) / 2
es = ['aa','ad','da','dd']
c = ['orange','b','r','g']
colors_e = []
for i in range(len(edges)):
    j = []
    for ii in range(4):
        j.append(Bs[ii][edges[i][0], edges[i][1]])
    colors_e.append(c[np.argmax(j)])
i = -1
for e in g.edges:
    i += 1
    ax.annotate("",
                xy=p_k[e[1]], xycoords='data',
                xytext=p_k[e[0]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color=colors_e[i],
                                shrinkA=5+counts[e[0]]/50,
                                shrinkB=5+counts[e[1]]/50,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=0.1",
                                mutation_scale=10, lw=weights[i],
                                ),
                )
ax.axis('off')
for i in range(len(edges)):
    if edges[i][0] == edges[i][1]:
        gap = counts[edges[i][0]] + 200
        center = tuple([p_k[edges[i][0]][0], p_k[edges[i][0]][1]+gap])
        l = mpl.patches.Arc(center,300,500,0,-60,240,
            lw=weights[i], color=colors_e[i])
        ax.add_patch(l)

x = ax.get_xlim()
y = ax.get_ylim()
ax.text(x[0] + np.ptp(x)/10, y[1] - np.ptp(y)/10, 'R', fontsize=30)
ax.text(x[1] - np.ptp(x)/10, y[1] - np.ptp(y)/10, 'L', fontsize=30)
for i in range(4):
    ax.text(x[1] - np.ptp(x)/20, y[0] + np.ptp(y)/40*i, es[i], c=c[i], fontsize=20)
ax.text(x[1] - np.ptp(x)/8, y[0] + np.ptp(y)/40*4, 'dominant edge type:', fontsize=20)
# plt.show()
plt.savefig('connectivity-anatomical_loc', transparent=False, facecolor='white', bbox_inches = "tight", dpi=300)

# %%
fig,ax = plt.subplots(1, figsize=(28,28))
ax.set_aspect('equal', adjustable='box')
g = nx.MultiDiGraph()
g.add_nodes_from(range(len(region_avg_locs)))
p_k = {}
for i in range(len(labels)):
    p_k[i] = region_avg_locs[i,:]
colors_n = []
for i in labels:
    colors_n.append(mode(meta[meta['hemi-class_4']==i]['color']))    
# nx.draw_networkx_nodes(g, pos=p_k, node_size=counts, node_color=colors_n, ax=ax)
ax = Axes3D(fig)
for key, value in p_k.items():
    xi = value[0]
    yi = value[1]
    zi = value[2]
    ax.scatter(xi, yi, zi, s=counts[key]*2.5, c=colors_n[key])


label_pos = p_k.copy()
offset_size = 500
for i in p_k.keys():
    label_pos[i] = [j+counts[i]*0.1+offset_size for j in p_k[i]]
label_pos[15][1] = label_pos[15][1] + offset_size
label_pos[16][1] = label_pos[16][1] + offset_size
node_labels = []
n = int((len(labels)-1) / 2)
l = np.unique(meta['class_4'])
for i in range(n+1):
    name = list(class_4.keys())[int(labels[i][1:])].split('_')[-1]
    node_labels.append(name)
for i in range(n+1, len(labels)):
    node_labels.append('')
node_labels_k = {}
for i in range(len(node_labels)):
    node_labels_k[i] = node_labels[i]
node_labels_k[0] = node_labels_k[0] + ' (C)'
node_labels_k[24] = node_labels_k[24] + ' (L)'
# nx.draw_networkx_labels(g, pos=label_pos, labels=node_labels_k, font_color='y', ax=ax)
for key, value in label_pos.items():
    xi = value[0]
    yi = value[1]
    zi = value[2]
    ax.text(xi, yi, zi, node_labels_k[key], c='y', fontsize=20)
# idx = np.where(np.array(p_val_bootstrap) == np.min(p_val_bootstrap))[0]
# edges = block_pairs[idx]
# edges = [tuple(edges[i,:]) for i in range(edges.shape[0])]
# g.add_edges_from(edges)
# weights = block_p.ravel()[idx]
# weights = (np.log(weights) - np.min(np.log(weights))) / 2
# es = ['aa','ad','da','dd']
# c = ['orange','b','r','g']
# colors_e = []
# for i in range(len(edges)):
#     j = []
#     for ii in range(4):
    #     j.append(Bs[ii][edges[i][0], edges[i][1]])
    # colors_e.append(c[np.argmax(j)])
# i = -1
# for e in g.edges:
#     i += 1
#     ax.annotate3D("",
#                 xyz=p_k[e[1]], xycoords='data',
#                 xytext=p_k[e[0]][1:], textcoords='data',
#                 arrowprops=dict(arrowstyle="->", color=colors_e[i],
#                                 shrinkA=5+counts[e[0]]/50,
#                                 shrinkB=5+counts[e[1]]/50,
#                                 patchA=None, patchB=None,
#                                 connectionstyle="arc3,rad=0.1",
#                                 mutation_scale=10, lw=weights[i],
#                                 ),
#                 )
# # ax.axis('off')
# # for i in range(len(edges)):
# #     if edges[i][0] == edges[i][1]:
# #         gap = counts[edges[i][0]] + 300
# #         center = tuple([p_k[edges[i][0]][0], p_k[edges[i][0]][1]+gap])
# #         l = mpl.patches.Arc(center,300,500,0,-60,240,
# #             lw=weights[i], color=colors_e[i])
# #         ax.add_patch(l)

# # x = ax.get_xlim()
# # y = ax.get_ylim()
# # ax.text(x[0] + np.ptp(x)/10, y[1] - np.ptp(y)/10, 'R')
# # ax.text(x[1] - np.ptp(x)/10, y[1] - np.ptp(y)/10, 'L')
# plt.show()
plt.savefig('3d_regions.pdf', transparent=False, facecolor='white', bbox_inches = "tight")

# %%
# from matplotlib.text import Annotation
# from mpl_toolkits.mplot3d.proj3d import proj_transform
# class Annotation3D(Annotation):

#     def __init__(self, text, xyz, *args, **kwargs):
#         super().__init__(text, xy=(0, 0), *args, **kwargs)
#         self._xyz = xyz

#     def draw(self, renderer):
#         x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
#         self.xy = (x2, y2)
#         super().draw(renderer)
        
# def _annotate3D(ax, text, xyz, *args, **kwargs):
#     '''Add anotation `text` to an `Axes3d` instance.'''

#     annotation = Annotation3D(text, xyz, *args, **kwargs)
#     ax.add_artist(annotation)

# setattr(Axes3D, 'annotate3D', _annotate3D)
# %%
