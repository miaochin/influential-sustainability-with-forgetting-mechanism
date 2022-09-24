"""
@author: Miao-Chin Yen
"""

from typing import List, Mapping, Union, Tuple
import random
import numpy as np

''' 
    In stochastic version of EIC, we would like to utilize the budget efficiently. Hence, we attempt 
    to find the best node for maximizing influential sustainability. Thus, we would select a node and keep activating it.
    node status: 'S' - node is inactive; 'I' - node is newly activated (NAN); 'R' - node is active but not newly activated
'''
    
class Graph:
    
    def __init__(self, G_edges, cycle_list, f_prob, seed_list, seed_repeat_times, simu_times):
        self.nodes: Mapping[str, List[str]] = {}
        self.edges: Mapping[str, List[Tuple[str, float]]] = G_edges
        self.cycle_list: List[List[str]] = cycle_list
        self.f_prob: float = f_prob
        self.seed_list : List[str] = seed_list
        self.seed_repeat_times: int = seed_repeat_times
        self.simu_times: int = simu_times
        
    def add_nodes(self, node_list: List[str]) -> Mapping[str, List[str]]:
        for n in node_list:
            self.nodes[n] = ['S']
            
        return self.nodes
        
    def propogation(self) -> str:
        
        display = ''
            
        ''' Note that we could only activate at most a seed at a specific timestamp'''       
        
        aggregate_cycle_times_sum: Mapping[str, Mapping[Tuple[str, ...], int]] = {} 

        for seed_element in self.seed_list:
            
            display += 'Seed: ' + str(seed_element) + '\n'
            
            ''' Initialize cycle_times_sum for different seed set''' 
            cycle_times_sum: Mapping[Tuple[str, ...], int] = {tuple(c): 0 for c in self.cycle_list} 
            
            '''
                sustain_round:  store the propagation sustain rounds for each simulation;
                node_sustain_times: dictionary used for calculating average rounds a node can sustain if it's activated among all simulations
            '''     
            sustain_round: List[int] = [] 
            node_sustain_times: Mapping[str, List[int]] = {} 
            simulation_times = self.simu_times
            
            while simulation_times > 0:  
                    
                ''' 
                    transition: record inactive -> active & active -> inactive time slot for each node 
                    seed_times_count: the remaining times we can seed ;
                    round: maximum number of rounds propgation can last;
                    cycle_freq: the number of times every cycle appears as a SC;
                    NAN_sum: store NAN set for each round;
                    Active_sum: store active set for each round
                ''' 
                
                transition: Mapping[str, List[int]] = {}  
                seed_times_count = self.seed_repeat_times - 1 
                round: int = 6000 
                iteration = round  
                cycle_freq: Mapping[Tuple[str, ...],int] = {} 
                NAN_sum: List[List[str]] = [list(seed_element)]  
                Active_sum : List[List[str]] = [list(seed_element)] 
                
                self.nodes[seed_element][0] = 'I'
                transition[seed_element] = [0]

                reseed = []
                
                while iteration > 0 :
                    
                    seed = [] # NAN for each round
                    for n in self.nodes:
                        if self.nodes[n][0] == 'I':
                            seed.append(n)
                            
                    ''' NAN for each round tries to activate its neighbors'''
                    for n in seed:
                        for i in range(len(self.edges[n])):
                            if self.nodes[self.edges[n][i][0]][0] == 'S':
                                sample = np.random.choice(2, 1, p = [1-self.edges[n][i][1],self.edges[n][i][1]])
                                if sample == 1:
                                    self.nodes[self.edges[n][i][0]][0] = 'I'
                                    if self.edges[n][i][0] not in transition:
                                        transition[self.edges[n][i][0]] = [round-iteration+1]
                                    else:   
                                        transition[self.edges[n][i][0]].append(round-iteration+1)
                        self.nodes[n][0] = 'R'
                    for n in self.nodes:
                        if  self.nodes[n][0] == 'R':
                            f_sample = np.random.choice(2, 1, p = [self.f_prob,1-self.f_prob])
                            if f_sample == 0:
                                self.nodes[n][0] = 'S'
                                transition[n].append(round-iteration+1)
                                
                    for n in reseed:
                        self.nodes[n][0] = 'I'
                        transition[n].append(round-iteration+1)
                        
                    reseed = []
                    iteration -= 1
                    
                    ''' Store NAN set and Active set to NAN_sum & Active_sum respectively '''
                    NAN = []
                    Active = []
                    for n in self.nodes:
                        if  self.nodes[n][0] == 'I': 
                            NAN.append(n)
                            Active.append(n)
                        if self.nodes[n][0] == 'R':
                            Active.append(n)
                    NAN_sum.append(NAN)
                    Active_sum.append(Active)
                    
                    ''' 
                        Check if all nodes are inactive and whether we need to reseed;
                        if we cannot activate seeds and all nodes are inactive, the propagation process ends
                    '''
                    count = 0
                    for n in self.nodes:
                        if self.nodes[n][0]=='S':
                            count += 1
                    if len(self.nodes)-count == 0 and seed_times_count > 0:
                        seed_times_count -= 1
                        reseed = list(seed)
                    elif len(self.nodes)-count == 0 and seed_times_count == 0:
                        break         
                '''
                    --------------------------------------------- propagation process ends ------------------------------------------
                ''' 
            
                ''' check # of SC for every cycle in the propagation process '''
                for x in range(len(self.cycle_list)):
                    
                    '''
                        cycle_times: each element is a tuple where
                            the first entry: cycle starts at round r
                            the second entry: the element in the cycle is NAN at round r
                    '''
                    cycle_times: List[Tuple[int,str]] = []
                    for i in range(len(NAN_sum)):
                        ''' Check if the propagation process lasts long for a cycle to be an SC'''
                        if i + len(self.cycle_list[x]) <= len(NAN_sum):
                            for j in NAN_sum[i]:
                                ''' Find if there is an NAN node to start for a cycle in propagation process'''
                                if j in self.cycle_list[x]:
                                    ''' Translate the cycle to a preferred order:  A -> B -> C -> A => B -> C -> A -> B '''
                                    reorder_cycle = self.cycle_list[x][self.cycle_list[x].index(j):len(self.cycle_list[x])]
                                    for k in range(self.cycle_list[x].index(j)):
                                        reorder_cycle.append(self.cycle_list[x][k])
                                    reorder_cycle.append(j)
                                    count_cycle = 0
                                    for k in range(len(self.cycle_list[x])):
                                        if reorder_cycle[k+1] in NAN_sum[i+k+1]:
                                            count_cycle += 1
                                        if reorder_cycle[k+1] not in NAN_sum[i+k+1]:
                                            break
                                    if count_cycle == len(self.cycle_list[x]):
                                        cycle_times.append((i,reorder_cycle[0]))
                                        ''' If the NAN has been calculated as part of stable cycle for some cycles, it cannot be calculated again'''
                                        for l in range(len(self.cycle_list[x])):
                                            NAN_sum[i+l].remove(reorder_cycle[l])

                    cycle_freq[tuple(self.cycle_list[x])]=len(cycle_times)
                    
                ''' 
                    node_sustain_times_round: used for calculating average rounds a node can sustain if it's activated;
                        element of node_sustain_times_round: (first, second)
                            first: total number of rounds a node is in active state during propgation process
                            second: total number of times a node transitions
                    ex: if 'A': [2, 3, 5, 7] which means inactive -> active at round 2, 5 ; active -> inactive at round 3, 7
                    => 'A': (3-2+7-5, 2) = (3, 2)
                '''
                node_sustain_times_round: Mapping[str, List[int]] = {}
                
                for c in cycle_times_sum:
                    cycle_times_sum[c] += cycle_freq[c]
                simulation_times -= 1  
                
                sustain_round.append(len(Active_sum) - self.seed_repeat_times)
                
                for i in transition:
                    node_sustain_times_round[i] = [0,0]
                    node_sustain_times_round[i][1] = int(len(transition[i])/2)
                    for j in range(0,len(transition[i]),2):
                        node_sustain_times_round[i][0] += transition[i][j+1] - transition[i][j]
                        
                for i in node_sustain_times_round:
                    if i not in node_sustain_times:
                        node_sustain_times[i] = node_sustain_times_round[i]
                    else:
                        node_sustain_times[i][0] += node_sustain_times_round[i][0]
                        node_sustain_times[i][1] += node_sustain_times_round[i][1]

            n_s_t_round: int = 0
            n_s_t_times: int = 0
            
            ''' average node sustain rounds calculation'''
            for i in node_sustain_times:
                n_s_t_round += node_sustain_times[i][0]
                n_s_t_times += node_sustain_times[i][1]

            display += '  Average node sustain rounds:' + str(n_s_t_round/n_s_t_times) + '\n'
            
            ''' average propagation sustain rounds calculation'''
            
            average_sustain_rounds: int  = sum(sustain_round)/len(sustain_round)
            
            display += '  Average propagation sustain rounds:' + str(average_sustain_rounds) + '\n'
            
            aggregate_cycle_times_sum[seed_element] = cycle_times_sum.copy()
        

        seed_cycle_times = {}
        
        for i in self.seed_list:
            seed_cycle_times[i] = 0
            for j in aggregate_cycle_times_sum[i]:
                seed_cycle_times[i] += aggregate_cycle_times_sum[i][j]
        
        display += str(aggregate_cycle_times_sum) + '\n'
        display += str(seed_cycle_times)
        
        print(display) 
        return display 


if __name__ == '__main__':
    
    f_prob: float = 0.02
    
    # seed_repeat_times: budget
    seed_repeat_times: int = 2
    simu_times: int = 12
    
    node_list: List[str] = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']
    
    G_edges: Mapping[str, List[Tuple[str, float]]] = {'A': [('G', 0.5), ('D', 0.4), ('R', 0.7)], 'B': [('V', 0.3), ('K', 0.1), ('I', 0.6)],'C': [('P', 0.4), ('O', 0.8), ('L', 0.5)], 'D': [('E', 0.7)], 'E': [('F', 0.7)], 'F': [('B', 0.45)], 'G': [('H', 0.9)], 'H': [('B', 0.5)], 'I': [('J', 0.7)], 'J': [('C', 0.7)], 'K': [('C', 1)], 'L': [('M', 0.2)], 'M': [('A', 0.6)], 'O': [('A', 0.9)], 'P': [('Q', 0.5)], 'Q': [('B', 0.9)], 'R': [('S', 0.83)],'S': [('T', 0.4)], 'T': [('C', 0.35)], 'U': [('A', 0.55)], 'V': [('U', 0.85)]}

    seed_list: List = ['A','B']
    
    cycle_list: List[List[str]] = [['A', 'R', 'S', 'T', 'C', 'O'],
            ['B', 'I', 'J', 'C', 'P', 'Q'], 
            ['A', 'G', 'H', 'B', 'I', 'J', 'C', 'O'],
            ['A', 'G', 'H', 'B', 'V', 'U'],
            ['A', 'D', 'E', 'F', 'B', 'I', 'J', 'C', 'O'],
            ['B', 'K', 'C', 'P', 'Q'], 
            ['A', 'G', 'H', 'B', 'K', 'C', 'O'], 
            ['A', 'D', 'E', 'F', 'B', 'V', 'U'], 
            ['A', 'D', 'E', 'F', 'B', 'K', 'C', 'O'],
            ['A', 'R', 'S', 'T', 'C', 'L', 'M'],
            ['A', 'G', 'H', 'B', 'I', 'J', 'C', 'L', 'M'],
            ['A', 'R', 'S', 'T', 'C', 'P', 'Q', 'B', 'V', 'U'],
            ['A', 'D', 'E', 'F', 'B', 'I', 'J', 'C', 'L', 'M'], 
            ['A', 'G', 'H', 'B', 'K', 'C', 'L', 'M'],
            ['A', 'D', 'E', 'F', 'B', 'K', 'C', 'L', 'M']]
        
    graph = Graph( G_edges, cycle_list, f_prob, seed_list, seed_repeat_times, simu_times)
    
    graph.add_nodes(node_list)

    graph.propogation()