import networkx as nx
import pandas as pd
import numpy as np
import os
import random
import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification
from tensorflow import keras
from sklearn.linear_model import LogisticRegression

data_dir = "data/"   # working directory
walk_length = 3
number_of_walks = 3
batch_size = 32
epochs = 50
layer_sizes = [16]

edgelist = pd.read_csv( os.path.join(data_dir, "pair_Sa_12"), 
    sep="\t", header=None, names=["source", "target", "label", "eo1", "eo2"])
file_node = pd.read_csv( os.path.join(data_dir, "pair_content_Sa_12"), sep="\t" )
nodes = file_node.set_index('ID')
feats = nodes.columns[0:(len(nodes.columns)-1)]

G_e_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
nx.set_node_attributes(G_e_nx, "label", "label")
G_e = sg.StellarGraph.from_networkx(G_e_nx, node_features=nodes[feats])

print(G_e.info())

edge_t = pd.read_csv( os.path.join(data_dir, "pair_Sa"), 
  sep="\t", header=None, names=["source", "target", "label", "eo1", "eo2"])
file_node_t = pd.read_csv( os.path.join(data_dir, "pair_content_Sa"), sep="\t" )
node_t = file_node_t.set_index('ID')

G_t_nx = nx.from_pandas_edgelist(edge_t, edge_attr="label")
nx.set_node_attributes(G_t_nx, "label", "label")
G_t = sg.StellarGraph.from_networkx(G_t_nx, node_features=node_t[feats])

print(G_t.info())

edge_p = pd.read_csv( os.path.join(data_dir, "pred/pair"),
    sep="\t", header=None, names=["source", "target", "label"] )
file_node_p = pd.read_csv( os.path.join(data_dir, "pred/content"), sep="\t")
node_p = file_node_p.set_index('ID')

G_p_nx = nx.from_pandas_edgelist(edge_p, edge_attr="label")
nx.set_node_attributes(G_p_nx, "label", "label")
G_p = sg.StellarGraph.from_networkx(G_p_nx, node_features=node_p[feats])

print(G_p.info())

unsupervised_samples = UnsupervisedSampler(
    G_e, nodes=list(G_e.nodes()), length=length, number_of_walks=number_of_walks
)
generator = Attri2VecLinkGenerator(G_e, batch_size)
attri2vec = Attri2Vec(layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None)
x_inp, x_out = attri2vec.in_out_tensors()
prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-2),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)
history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
)

x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_gen = Attri2VecNodeGenerator(G_t, batch_size).flow(node_t.index)
emb = embedding_model.predict(node_gen, workers=4, verbose=1)
node_p_gen = Attri2VecNodeGenerator(G_p, batch_size).flow(node_p.index)
emb_p = embedding_model.predict(node_p_gen, workers=4, verbose=1)

### average operator
pred_feat1 = []; pred_feat2 = []; 
lbl1 = []; lbl2 = []; lbl_p = [];
for i in range(len(edge_t)):
    n1 = np.where( node_t.index == edge_t["source"][i] )[0].tolist()
    n2 = np.where( node_t.index == edge_t["target"][i] )[0].tolist()
    pred_feat1.append( np.ravel( (emb[n1] + emb[n2] ) / 2 ) )

for i in range(len(edge_p)):
    n1 = np.where( node_p.index == edge_p["source"][i] )[0].tolist()
    n2 = np.where( node_p.index == edge_p["target"][i] )[0].tolist()
    pred_feat2.append( np.ravel( (emb_p[n1] + emb_p[n2] ) / 2 ) )
    lbl_p.append( edge_p["label"][i] )

clf_edge_pred_from_feat = LogisticRegression(verbose=0, solver="lbfgs", 
    class_weight='balanced', max_iter=500, multi_class='multinomial')
clf_edge_pred_from_feat.fit(pred_feat1, edge_t["label"])
edge_pred_from_feat = clf_edge_pred_from_feat.predict_proba(pred_feat2)
positive_class_index = 0
negative_class_index = 1
if clf_edge_pred_from_feat.classes_[1] == 1:
    positive_class_index = 1
elif clf_edge_pred_from_feat.classes_[2] == 1:
    positive_class_index = 2

if clf_edge_pred_from_feat.classes_[0] == 2:
    negative_class_index = 0
elif clf_edge_pred_from_feat.classes_[2] == 2:
    negative_class_index = 2

np.savetxt( os.path.join("output_prob.txt"),
     np.stack([edge_pred_from_feat[:, positive_class_index],
     edge_pred_from_feat[:, negative_class_index], lbl_p], 1) )


