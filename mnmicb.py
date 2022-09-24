"""
@author: Miao-Chin Yen
"""

import numpy as np
import random
import math
from math import gcd
from typing import Dict, List, Mapping, Union


'''
    ---------------------------General Variables-----------------------------
    G_edges: dictionary of edges structure in the graph
    s: source node
    d: destination node
    multi_in: nodes with multiple in-neighbors
    k/f_avg: forgetting round
    l: cycle length
    max_cycle_len: maximum cycle length
    multi_hop: hops of nodes in MI-N and covered within c_l from starting node of c_l
        e.g.  c_l = ('H', 'B', 'K', 'C', 'L', 'M', 'A', 'G') and 'B', 'C', 'A' are in MI-N -> multi_hop = {'B': 1, 'C': 3, 'A': 6}
    multi_hop_set: reference function - multi_hop_set
    multi_pair_distance: len(A_i->A_j)  for every pair of (A_i, A_j)

'''


''' allpath: find all paths from s to d without nodes in not_in '''

def allpath(G_edges,s,d,visited,path,Path,not_in):
    ''' 
        visited: dictionary for memorizing if a node is visited; default: {}
        path: for constructing path from s to d
        Path: set for storing all paths from s to d; default: []
        not_in: nodes not allowed to be visited when finding path from s to d
    ''' 
    visited[s]= True
    path.append(s)
    if s == d: 
        Path.append(path.copy())
    else: 
        for i in G_edges[s]: 
            if (visited[i] == False) and (i not in not_in): 
                allpath(G_edges,i,d,visited,path,Path,not_in) 
    path.pop() 
    visited[s]= False
    return Path


''' BFS: bfs algorithm for specific node s & return the nodes can be visited when the source node is s'''

def BFS(G_edges,s):
    level = {}
    parent = {s: None}
    i = 1
    range_s = [] 
    frontier = [s]      
    while frontier:
        next = [ ]     
        for u in frontier:
            for v in G_edges [u]:
                if v not in level: 
                    level[v] = i
                    parent[v] = u
                    next.append(v)
                    range_s.append(v)
        frontier = next
        i += 1
    return range_s
    

''' 
    hop: calculate hops for nodes within path s->d with s being h hops from a target node
    (Note that for path s->d, only s and d are allowed to be in MI-N)
'''

def hop(G_edges,s,d,h,multi_in):
    ''' 
        h: s is h hops from a target node
    ''' 
    G_nodes_hop = {}
    for i in G_edges:
        G_nodes_hop[i] = []
    G_nodes_hop[s] = [h] 
    parent = [s]
    successor = []
    i = h + 1
    while len(parent) > 0:
        for v in parent:
            for u in G_edges[v]:
                if (u!= d) and (u not in multi_in):
                    G_nodes_hop[u].append(i)
                    successor.append(u)
                if u == d:
                    G_nodes_hop[u].append(i)
        parent = successor 
        successor = []
        i += 1 
    return G_nodes_hop



''' distance: construct len(A_i->A_j)  for every pair of (A_i, A_j)'''

def distance(G_edges, multi_in):
    dist = {}
    for i in multi_in:
        for j in multi_in:
            if  (j in BFS(G_edges,i)) and j not in dist:
                dist[j] = {i: hop(G_edges,i,j,0,multi_in)[j]}
            if  (j in BFS(G_edges,i)):
                dist[j][i] = hop(G_edges,i,j,0,multi_in)[j]
    dist_c = {}
    # delete the empty dist[i][j]
    for i in dist:
        for j in dist[i]:
            if len(dist[i][j]) > 0 and (i not in dist_c):
                dist_c[i] = {j: dist[i][j]}
            if len(dist[i][j]) > 0:
                dist_c[i][j] = dist[i][j]
    return dist_c 


''' multi_hop_set: construct possible hops for nodes in multi-hop '''

def multi_hop_set(multi_hop,multi_pair_distance):
    multi_hop_set = {}
    for i in multi_hop:
        multi_hop_set[i] = []
    for i in multi_hop_set:
        if i in multi_pair_distance:
            for j in multi_pair_distance[i]:
                if j in multi_hop:
                    for u in multi_pair_distance[i][j]:
                        b = u + multi_hop[j]
                        multi_hop_set[i].append(b)
    for i in multi_hop_set:
        multi_hop_set[i].append(multi_hop[i])
    return multi_hop_set 
    

''' mu_add: SLC condition for mu > h_i '''

def mu_add(k,l,h_i,mu):
    flag = False
    x = 0
    n = int(l/(k+1))
    m = int(l%(k+1))
    while (k+1)*x+h_i+int(x*m/n) <= mu:
        x += 1
    if (x-1) * m % n != 0:
        if (mu != h_i+(x-1)*(k+1)+int((x-1)*m/n)) and (mu < (k+1)*x+h_i+int((x-1)*m/n) or mu >= (k+1)*x+h_i+int(x*m/n)):
            flag = True
    else:
        if (mu < (k+1)*x+h_i+int((x-1)*m/n) or mu >= (k+1)*x+h_i+int(x*m/n)):
            flag = True
    return flag 


''' mu_minus: SLC condition for mu < h_i'''

def mu_minus(k,l,h_i,mu):
    flag = False
    x = 0
    n = int(l/(k+1))
    m = int(l%(k+1))
    while h_i-(k+1)*x-int(x*m/n) > mu:
        x += 1
    if (x * m) % n != 0:
        if (mu < (k+1)*(1-x)+h_i-int(x*m/n)-1 or mu >= (k+1)*(1-x)+h_i-int((x-1)*m/n)):
            flag = True
    else:
        if (mu < (k+1)*(1-x)+h_i-int(x*m/n) or mu >= (k+1)*(1-x)+h_i-int((x-1)*m/n)):
            flag = True
    return flag   
        
    
''' check_sl: check SLC conditions are satisfied'''
    
def check_sl(multi_hop,multi_hop_set,k,l):
    summation = 0
    count = 0
    for i in multi_hop_set:
        summation += len(multi_hop_set[i])
    for i in multi_hop_set:
        for j in multi_hop_set[i]:
            if j <= multi_hop[i]:
                x = mu_minus(k,l,multi_hop[i],j)
                if x == True:
                    count += 1
                else:
                    break
            if j > multi_hop[i]:
                x = mu_add(k,l,multi_hop[i],j)
                if x == True:
                    count += 1
                else:
                    break
    if count == summation:
        return True
    else:
        return False

''' dist: construct AIV and selected seeds '''

def dist(l, k ,max_cycle_len):
    n_l = int(l / (k+1))
    m_l = int(l % (k+1))
    
    # m_l = w * n_l + r_l
    w = int(m_l / n_l)
    r_l = int(m_l % n_l)
    
    #  construct x^{n_l,m_l} vector
    x = []
    divisor = gcd(n_l,m_l)
    
    #  gcd (n_l, m_l) needs to be 1 
    if divisor != 1:
            n_l = int(n_l / divisor)
            m_l = int(m_l / divisor)
            w = int(m_l / n_l)
            r_l = int(m_l % n_l)
    if r_l == 0:
        for i in range(n_l):
            x.append(0)
    else:
        #  if condition: zeros are more; else condition: ones are more 
        if n_l - r_l > r_l:
            for i in range(n_l):
                x.append(0)
            if r_l > 0:
                x[0]=1
            if r_l > 1:
                x[int(n_l / r_l)] = 1
            if r_l > 2:
                x[1 + 2 * int(n_l / r_l)] = 1
            for i in range(r_l - 3):
                x[1 + int(n_l / r_l)+int((i + 2) * n_l/r_l)] = 1
        else:
            for i in range(n_l):
                x.append(1)
            for i in range(n_l - r_l):
                x[int((i + 1) * n_l / (n_l - r_l)) - 1] = 0
                
    #  construct AIV 
    for i in range(n_l):
        x[i] += w

    #  Seed set: first round of SL 
    S = []
    s = 0
    j = n_l - 1
    while s<= max_cycle_len - 1:
        S.append(s)
        s += x[j] + k + 1 
        j -= 1
        if j < 0:
            j = n_l - 1
    return S
    
    
''' loop_len: calculate SL rounds '''

def loop_len(l,k):
    n_l = int(l / (k+1))
    m_l = int(l % (k+1))
    divisor = gcd(n_l,m_l)
    if divisor != 1:
        n_l = int(n_l/divisor)
        m_l = int(m_l/divisor)
    return int((k + 1) * n_l + m_l)
    
    
''' loop: construct sustainable loop (SL)'''
    
def loop(l,k,max_cycle_len):
    sl_round = loop_len(l,k)
    SL = []
    S = dist(l,k,max_cycle_len)
    for i in range(sl_round):
        round = [x+i for x in S]
        SL.append(round)
    return SL


''' seed_set: choose the seed set  '''
def seed(multi_hop,multi_hop_set,multi_in,c_l):
    '''
        c_l: cycle 
    '''
    seed_set = []
    for i in range(len(c_l)):
        a = multi_in[ : ]
        c = multi_hop_set.copy()
        # b: rewrite cycle c_l
        if i == 0:
            b = c_l[i : len(c_l)]
        if i > 0:
            b = c_l[i : len(c_l)]
            for j in range(i):
                b.append(c_l[j])
        for j in multi_hop:
            if multi_hop[j] in b[0]:
                a.remove(j)
                del c[j]
        hop = {}
        for j in a:
            hop[j] = []
        for j in range(len(b)-1):
            for u in c:
                d = []
                for k in b[j+1]:
                    if k in c[u]:
                        d.append(k)
                hop[u].append(d)
        hop_f = {}
        for j in hop:
            hop_f[j] = []
        for j in hop:
            for u in hop[j]:
                if len(u) > 0:
                    hop_f[j].append(u)
        count = 0
        for j in a:
            if multi_hop[j] not in hop_f[j][0]:
                count += 1
        if count == 0:
            seed_set.append(c_l[i])
    return seed_set 
    

''' enumer: construct c_lxp_l and check if cycle can be a PSC and thus formulate SL'''

def enumer(G_edges,visited,undetermined,determined,multi_hop,a,g,multi_pair_distance,leng,multi_in,f_avg,max_length,cycle_seed,cycle):
    '''
        visited : dictionary for checking if an edge is visited
        undetermined: nodes in MI-N not covered in c_l
        determined: nodes in MI-N covered in c_l
        a & g: variable for storing data; default: [] 
        leng : length of c_l
        f_avg: forgetting round
        max_length: possilbe maximum cycle length
        cycle_seed: dictionary of PSC and corresponding seeds selection
        cycle: target c_l
    '''
    ''' if c_l covers all nodes in MI-N '''
    if len(undetermined) == 0:
        g.append(multi_hop)
        if multi_hop not in a:
            a.append(multi_hop)
            p = multi_hop_set(multi_hop,multi_pair_distance)
            l = loop(leng,f_avg,max_length) 
            length = loop_len(leng,f_avg)
            if check_sl(multi_hop,p,f_avg,length) == True:
                if len(seed(multi_hop,p,multi_in,l)) > 0:
                    whole_hop = {}
                    for v in multi_hop:
                        whole_hop[v]=[]
                        whole_hop[v].append(multi_hop[v])
                    for v in multi_in:
                        for w in multi_in:
                            s = hop(G_edges,v,w,whole_hop[v][0],multi_in)
                            for y in s:
                                if y not in whole_hop and len(s[y])>0:
                                    whole_hop[y] = s[y]
                    seed_set=[]
                    for v in whole_hop:
                        if whole_hop[v][0] in seed(multi_hop,p,multi_in,l)[0]:
                            seed_set.append(v)
                    if cycle not in cycle_seed:
                        cycle_seed[cycle] = []
                        cycle_seed[cycle].append(seed_set)
                    else:
                        cycle_seed[cycle].append(seed_set)
    else:
        i = undetermined[0]
        for j in determined:
            b = determined.copy()
            b.remove(j)
            for k in allpath(G_edges,j,i,visited,[],[],b):
                d = determined.copy()
                ud = undetermined.copy()
                m_k = multi_hop.copy()
                c = []
                for u in range(len(k)):
                    if k[u] in undetermined:
                        m_k[k[u]] = m_k[j] + u
                        c.append(k[u])
                for m in c:
                    d.append(m)
                    ud.remove(m)
                enumer(G_edges,visited,ud,d,m_k,a,g,multi_pair_distance,leng,multi_in,f_avg,max_length,cycle_seed,cycle)
    return a



if __name__ == '__main__':
    
    G_edges={'A':['G','D','R'],'B':['V','K','I'],'C':['P','O','L'],'D':['E']
    ,'E':['F'],'F':['B'],'G':['H'],'H':['B'],'I':['J']
    ,'J':['C'],'K':['C'] ,'L':['M'],'M':['A'],'O':['A']
    ,'P':['Q'],'Q':['B'],'R':['S'],'S':['T'],'T':['C'],'U':['A'],'V':['U']}
    
    k = 3

    cycle = [['A', 'G', 'H', 'B', 'K', 'C', 'O'], ['A', 'G', 'H', 'B', 'K', 'C', 'L', 'M'],
        ['A', 'G', 'H', 'B', 'V', 'U'], ['A', 'G', 'H', 'B', 'I', 'J', 'C', 'O'], 
        ['A', 'G', 'H', 'B', 'I', 'J', 'C', 'L', 'M'], ['A', 'R', 'S', 'T', 'C', 'O'], 
        ['A', 'R', 'S', 'T', 'C', 'P', 'Q', 'B', 'V', 'U'], ['A', 'R', 'S', 'T', 'C', 'L', 'M'],
        ['A', 'D', 'E', 'F', 'B', 'K', 'C', 'O'], ['A', 'D', 'E', 'F', 'B', 'K', 'C', 'L', 'M'], 
        ['A', 'D', 'E', 'F', 'B', 'V', 'U'], ['A', 'D', 'E', 'F', 'B', 'I', 'J', 'C', 'O'],
        ['A', 'D', 'E', 'F', 'B', 'I', 'J', 'C', 'L', 'M'], ['B', 'K', 'C', 'P', 'Q'], 
        ['B', 'I', 'J', 'C', 'P', 'Q']]
        
    ''' construct dictionary of in-neighbors for every node '''
    G_nodes_innb = {}
    for i in G_edges:
        G_nodes_innb[i] = []
    for i in G_edges:
        for j in range(len(G_edges[i])):
            G_nodes_innb[G_edges[i][j]].append(i)

    # possible maximum cycle length
    max_length = len(G_edges)

    ''' construct MI-N'''
    multi_in = []
    for i in G_nodes_innb:
        if len(G_nodes_innb[i]) > 1:
            multi_in.append(i)
            
    multi_pair_distance = distance(G_edges,multi_in)    

    cycle_seed = {}
    for i in cycle:
        multi_hop = {}
        visited = {}
        for j in G_edges:
            visited[j] = False
        determined = []
        undetermined = multi_in.copy()
        for j in range(len(i)):
            if i[j] in multi_in:
                multi_hop[i[j]] = j
                determined.append(i[j])
                undetermined.remove(i[j])    
        enumer(G_edges,visited,undetermined,determined,multi_hop,[],[],multi_pair_distance,len(i),multi_in,k,max_length,cycle_seed,tuple(i))
        
    print(cycle_seed)