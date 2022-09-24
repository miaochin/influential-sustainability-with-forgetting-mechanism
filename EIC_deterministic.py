"""
@author: Miao-Chin Yen
"""

from typing import List, Mapping, Union
from unicodedata import name

''' 
    propagate_round: rounds to observe
    node status: 'S' - node is inactive; 'I' - node is newly activated (NAN); 'R' - node is active but not newly activated
'''

class Graph:
    
    def __init__(self, f_avg, propagate_round, seed):
        self.nodes: Mapping[str, List[Union[int, str]]] = {}
        self.edges: Mapping[str, List[str]] = {}
        self.f_avg: int = f_avg - 1
        self.propagate_round: int = propagate_round
        self.seed : List = seed
        
    def add_nodes(self, node_list: List) -> Mapping[str, List[Union[int, str]]]:
        for n in node_list:
            self.nodes[n] = [self.f_avg, 'S']  
            
        return self.nodes
        
    def add_edges(self, edge_dict: Mapping[str, List[str]]) -> Mapping[str, List[str]]:
        for source in edge_dict:
            if source not in self.nodes:
                self.add_nodes(source)
            if source not in self.edges:
                self.edges[source] = edge_dict[source]
            else:
                for edge in edge_dict[source]:
                    self.edges[source].append(edge)
        
        return self.edges
        
    def propogation(self) -> str:
        NAN: List = []
        for s in self.seed:
            self.nodes[s]=[self.f_avg,'I']
            NAN.append(s)
            
        Inac: List = []
        for n in self.nodes:
            if self.nodes[n][1] == 'S':
                Inac.append(n)
            
        display = "0:" + '\n'
        display += 'NAN:' + str(NAN) + '\n'
        display += '# of NAN:' + str(len(NAN)) + '\n'
        display += 'Inactive:' + str(Inac) + '\n'
        display += '# of Inactive:' + str(len(Inac)) + '\n'
        
        round = self.propagate_round
        while round > 0 :    
            '''  Propagation starts  '''
            for n in NAN:
                if self.nodes[n][1]=='I':
                    for i in range(len(self.edges[n])):
                        if self.nodes[self.edges[n][i]][1] == 'S':
                            self.nodes[self.edges[n][i]][1] = 'I'
                    self.nodes[n][1] = 'R'
            
            ''' 
                Forgetting round countdown and 
                tackle active nodes to become inactive nodes because of the forgetting mechanism
            '''
            for n in self.nodes:
                if self.nodes[n][1] == 'R':
                    if self.nodes[n][0] > 0:
                        self.nodes[n][0] -= 1
                    elif self.nodes[n][0] == 0:
                        self.nodes[n] = [self.f_avg,'S']
                                    
            display += str(self.propagate_round - round + 1) + ':\n'
        
            ''' Formulate and count # of NAN and Inactive nodes '''
    
            NAN = []
            for n in self.nodes:
                if  self.nodes[n][1] == 'I':
                    NAN.append(n)

            Inac = []
            for n in self.nodes:
                if self.nodes[n][1] == 'S':
                    Inac.append(n)

            display += 'NAN:' + str(NAN) + '\n'
            display += '# of NAN:' + str(len(NAN)) + '\n'
            display += 'Inactive' + str(Inac) + '\n'
            display += '# of Inactive:' + str(len(Inac)) + '\n'
            
            round -= 1
            
            '''Propagtion stops ''' 
            if len(Inac) == len(self.nodes):
                display += str(self.propagate_round - round - 1) + '\n'
                display += 'propagation ends'
                break
            
        print(display)
        return display




if __name__ == '__main__':
    
    f_avg: int = 5
    propagate_round : int = 17
    node_list: List = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']
    edge_dict: Mapping = {'A':['G','D','R'],'B':['V','K','I'],'C':['P','O','L'],'D':['E'],'E':['F'],'F':['B'],'G':['H'],'H':['B'],'I':['J'],'J':['C'],'K':['C'] ,'L':['M'],'M':['A'],'O':['A'],'P':['Q'],'Q':['B'],'R':['S'],'S':['T'],'T':['C'],'U':['A'],'V':['U']}
    seed: List = ['A','B','M','Q']
    
    graph = Graph(f_avg, propagate_round, seed)
    
    graph.add_nodes(node_list)
    graph.add_edges(edge_dict)
    
    graph.propogation()