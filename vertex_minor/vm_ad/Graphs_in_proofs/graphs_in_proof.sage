# This file builds a dictionary containing all the graphs considered in the proof of Theorem 4.3 and 4.5 in https://arxiv.org/abs/1805.05306 and then checks that these are not distance-hereditary

load("GraphClasses.sage")

G_dict={}

# Theorem 4.3

G_dict43={}

# Step 2

G=SimpleGraph({0:[4,1],1:[0,4,2,3],2:[3,1],3:[4,2,1],4:[0,1,3]})
G_dict43[2]=[G]

# Step 3

G_dict43[3]=[]
G=SimpleGraph({0:[4,1,2],1:[0,5],2:[0,3],3:[4,2],4:[5,3,0],5:[4,1]})
edges=[(0,3),(0,5),(3,5)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict43[3].append(Gtmp)

# Step 5

G=SimpleGraph({0:[4,1],1:[0,4,2,3],2:[3,1],3:[4,2,1],4:[0,1,3]})
G_dict43[5]=[G]

G_dict["Thm43"]=G_dict43

# Theorem 4.5

G_dict45={}

# Step 1

G_dict45[1]=[]
G=SimpleGraph({0:[1,2,3,5,4],1:[2,3,0],2:[5,3,1,0],3:[4,2,1,0],4:[3,0],5:[2,0]})
edges=[(4,5)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict45[1].append(Gtmp)

# Step 3

# Case 1

G=SimpleGraph({0:[4,1],1:[0,4,2,3],2:[3,1],3:[4,2,1],4:[0,1,3]})
G_dict45[3]=[G]

# Case 2

# Subcase 1

G=SimpleGraph({0:[2,1],1:[4,2,0],2:[1,0,3],3:[2,4],4:[1,3]})
G_dict45[3].append(G)

# Subcase 2

G=SimpleGraph({0:[3,1],1:[4,3,2,0],2:[3,4,1],3:[4,2,1,5,0],4:[3,2,1,5],5:[4,3]})
edges=[(0,5)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict45[3].append(Gtmp)

# Subcase 3

# Subsubcase 1

G=SimpleGraph({0:[2,1],1:[4,2,0],2:[1,0,3],3:[2,4],4:[1,3]})
G_dict45[3].append(G)

# Subsubcase 2

G=SimpleGraph({0:[1,2,3,5,4,7,6],1:[0,2,3,5],2:[1,3,0,4],3:[2,1,0,5,4],4:[0,5,2,3,7],5:[0,4,1,3,6],6:[5,0],7:[4,0]})
edges=[(6,7),(6,4),(6,2),(7,5),(7,1)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict45[3].append(Gtmp)

# Step 5

# Case 1

G_dict45[5]=[]
G=SimpleGraph({0:[3,1],1:[4,3,2,0],2:[3,4,1],3:[4,2,1,5,0],4:[3,2,1,5],5:[4,3]})
edges=[(0,5)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict45[5].append(Gtmp)

# Case 2

# Subcase 1

G=SimpleGraph({0:[1,2,3,5,4],1:[2,3,0],2:[5,3,1,0],3:[4,2,1,0],4:[3,0],5:[2,0]})
edges=[(4,5)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict45[5].append(Gtmp)

# Subcase 2

G=SimpleGraph({0:[1,2,3,4,5,6,7],1:[0,2,3,5,4,7],2:[0,1,3,4],3:[0,1,2,4,5],4:[0,2,3,7,1],5:[0,1,3,6],6:[0,5],7:[0,4,1]})
edges=[(6,7),(6,4),(6,2),(7,5),(4,5)]
for es in Subsets(edges):
    Gtmp=SimpleGraph(G)
    Gtmp.add_edges(es)
    G_dict45[5].append(Gtmp)

G_dict["Thm45"]=G_dict45


####################################################
# Check that all graphs are not distance-hereditary
####################################################


fail=False
for Thm_dict in G_dict.values():
    for G_list in Thm_dict.values():
        for G in G_list:
            if G.is_DH():
                print("Graph for {} at step {} is distance-hereditary!")
                fail=True
if not fail:
    print("All graphs in Theorems 4.3 and 4.5 are not distance-hereditary.")
