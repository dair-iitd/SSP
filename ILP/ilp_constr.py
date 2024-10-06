import numpy as np
import pandas  as pd
#from sentence_transformers import util
#from utils import *
from tqdm import tqdm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpConstraint, LpConstraintGE, LpConstraintEQ,LpConstraintLE, GUROBI_CMD, GLPK_CMD
import pdb
import re

import gurobipy as grb

def indic_pos(x):
    if x.item() > 0:
        return 1
    else:
        return 0

class SEER:
    '''
    The SEER algorithm.
    Attributes:
        k (int) : the number of nearest neighbor to filter
        M (int) : the maximum number of exemplars to select
        alpha (float) : the share of exemplars that should possess the attribute of the test instance
        beta (float) : the share of exemplars that should not possess the attribute of the test instance
        modules (list) : list of constraint modules to include
    '''
    def __init__(self,
                 k=100,
                 M=8,
                 taus={},
                 num_it=1,
                 betas=[2, 2, 2, 2, 0],
                 labels=['PER', 'LOC', "ORG", "DATE", "O"],
                 perc='80',
                 type='prob'):  
        
        self.k = k
        self.M = M
        #self.alpha = alpha
        assert len(betas) == 3 and len(labels) == 3
        self.betas = betas
        self.labels = labels
        self.taus = taus
        self.num_it = num_it
        self.type = type
        self.perc = perc


    def get_KNN(self,test_idx,similarity_matrix,train_filter=[]):
        '''
        Retrieves the nearest neighbor of a test instance given a similarity matrix.
        '''
        instance_similarity = similarity_matrix[test_idx]
        candidates = [idx for idx in np.argsort(instance_similarity) if idx in train_filter][-self.k:][::-1]  #First is the best one 
        return candidates
    
    def get_knapsack(self,
                     test_idx,
                     candidates_idx,
                     test_dataframe,
                     similarity_matrix,
                     name='seer knapsack problem'):
        problem = LpProblem(name,LpMaximize)
        candidates = LpVariable.dicts("instance",candidates_idx,0,cat=LpBinary)
        #Add objective
        problem += lpSum([similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx])        #feed similarity matrix
        #Add capacity constraints
        #problem += lpSum([train_dataframe.loc[i,'token_prompt_length']*candidates[i] for i in candidates_idx]) <= max_prompt_length   # Not needed
        #problem += lpSum([candidates[i] for i in candidates_idx]) == self.M   #Changed to equality (original was less than equal to)
        # Add Confidence constraints
        #problem += lpSum([test_dataframe.loc[i,'conf']*candidates[i]for i in candidates_idx]) >= self.tau * self.M
        #for idx in candidates_idx:
        #    #problem += lpSum([candidates[i]*(self.tau-test_dataframe.loc[i,'conf']) for i in [idx]]) <= 0
        #    problem += lpSum([candidates[i]*(test_dataframe.loc[i,self.taus]-test_dataframe.loc[i,'conf']) for i in [idx]]) <= 0
        #Add diversity constraints     (design in terms of label coverage)
        if self.num_it == 1:
            # for beta_,label in zip(self.betas, self.labels):
            #     problem += lpSum([test_dataframe.loc[i, label]*candidates[i] for i in candidates_idx]) > self.M * beta_   
            constraints_new = {0 : problem.addConstraint(
            LpConstraint(
            e=lpSum(candidates[i] for i in candidates_idx),
            sense=LpConstraintEQ,
            rhs=self.M,
            name="Eq_M"))
            }

            set_J = range(1, len(self.labels)+1)
            constraints = {j : problem.addConstraint(
            LpConstraint(
            e=lpSum(test_dataframe.loc[i, label+"_cnt"]*candidates[i] for i in candidates_idx),
            sense=LpConstraintGE,
            rhs=1,
            name="constraint_{0}".format(j)))
            for j,label in zip(set_J, self.labels)}

            constraints_new2 = []
            for j,label in enumerate(self.labels):
                for k, idx in enumerate(candidates_idx):
                    constraints_new2.append({len(self.labels)+1+k+j*len(candidates_idx): problem.addConstraint(
                    LpConstraint(
                    e= lpSum(candidates[i]*(self.taus[label+'_'+self.perc+'_PERC_PROB']-test_dataframe.loc[i,label+'_probs']) for i in [idx]),
                    sense=LpConstraintLE,
                    rhs=0.0,
                    name="constraint1_{0}_{1}".format(idx,j)
                    )
                    )
                    })
        #pdb.set_trace()
        return problem
    
    def solve_knapsack(self,problem,timelimit=5.0):
        #solver = GUROBI_CMD(timeLimit=timelimit)
        solver = GLPK_CMD()
        #pdb.set_trace()
        problem.solve()
        try:
            solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables() ],[v.varValue for v in problem.variables() ]))
        except:
            solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables()[1:]],[v.varValue for v in problem.variables()[1:]]))
        return solution
    
    def get_few_shot_exemplars(self,test_idx,similarity_matrix,test_dataframe,rem_lab_cons=False, rem_conf_cons=False, num_it=1):
        #candidates = self.get_KNN(test_idx,similarity_matrix,train_filter)   #Can do filtering if have lots of candidates
        if num_it == 1:
            candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
            #pdb.set_trace()
            problem = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix)
            solution = self.solve_knapsack(problem)
            
            selection = [k for k,v in solution.items() if v==1]
            few_shot_selection = [idx for idx in np.argsort(similarity_matrix[test_idx]) if idx in selection][::-1]
            #pdb.set_trace()
            curr_len = len(few_shot_selection)
            #pdb.set_trace()
            if curr_len < self.M:
                for _ in range(self.M-curr_len):
                    for idx in np.argsort(similarity_matrix[test_idx])[::-1]:
                        if idx not in few_shot_selection and idx != test_idx:
                            few_shot_selection.append(idx)
                            break
            assert len(few_shot_selection) == self.M
        else:
            assert self.M == 1
            few_shot_selection = []
            for j in range(num_it):
                if j == 0:
                    candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
                else:
                    candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx and j not in few_shot_selection]
                problem = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix)
                solution = self.solve_knapsack(problem)
        
                selection = [k for k,v in solution.items() if v==1]
                #pdb.set_trace()
                #assert len(selection) == 1
                few_shot_selection.append(selection[0])

        return few_shot_selection+[-1]*(self.M-len(few_shot_selection)), len(few_shot_selection)

# class SEER_SEQ:
#     '''
#     The SEER algorithm.
#     Attributes:
#         k (int) : the number of nearest neighbor to filter
#         M (int) : the maximum number of exemplars to select
#         alpha (float) : the share of exemplars that should possess the attribute of the test instance
#         beta (float) : the share of exemplars that should not possess the attribute of the test instance
#         modules (list) : list of constraint modules to include
#     '''
#     def __init__(self,
#                  k=100,
#                  M=8,
#                  taus={},
#                  num_it=1,
#                  betas=[2, 2, 2, 2, 0],
#                  labels=['PER', 'LOC', "ORG", "DATE", "O"],
#                  type='entr'):  
#         self.k = k
#         self.M = M
#         #self.alpha = alpha
#         self.betas = betas
#         self.labels = labels
#         self.taus = taus
#         self.num_it = num_it
#         self.type = type

#     def get_KNN(self,test_idx,similarity_matrix,train_filter=[]):
#         '''
#         Retrieves the nearest neighbor of a test instance given a similarity matrix.
#         '''
#         instance_similarity = similarity_matrix[test_idx]
#         candidates = [idx for idx in np.argsort(instance_similarity) if idx in train_filter][-self.k:][::-1]  #First is the best one 
#         return candidates
    
#     def get_knapsack(self,
#                      test_idx,
#                      candidates_idx,
#                      test_dataframe,
#                      similarity_matrix,
#                      name='seer knapsack problem'):
#         problem = LpProblem(name,LpMaximize)
#         candidates = LpVariable.dicts("instance",candidates_idx,0,cat=LpBinary)
#         #Add objective
#         problem += lpSum([similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx])        #feed similarity matrix
#         #Add capacity constraints
#         #problem += lpSum([train_dataframe.loc[i,'token_prompt_length']*candidates[i] for i in candidates_idx]) <= max_prompt_length   # Not needed
#         #problem += lpSum([candidates[i] for i in candidates_idx]) == self.M   #Changed to equality (original was less than equal to)
#         # Add Confidence constraints
#         #problem += lpSum([test_dataframe.loc[i,'conf']*candidates[i]for i in candidates_idx]) >= self.tau * self.M
#         #for idx in candidates_idx:
#         #    #problem += lpSum([candidates[i]*(self.tau-test_dataframe.loc[i,'conf']) for i in [idx]]) <= 0
#         #    problem += lpSum([candidates[i]*(test_dataframe.loc[i,self.taus]-test_dataframe.loc[i,'conf']) for i in [idx]]) <= 0
#         #Add diversity constraints     (design in terms of label coverage)
#         if self.num_it == 1:
#             # for beta_,label in zip(self.betas, self.labels):
#             #     problem += lpSum([test_dataframe.loc[i, label]*candidates[i] for i in candidates_idx]) > self.M * beta_   
#             set_J = range(1, len(self.labels)+1)
#             constraints = {j : problem.addConstraint(
#             LpConstraint(
#             e=lpSum(test_dataframe.loc[i, label+"_cnt"]*candidates[i] for i in candidates_idx),
#             sense=LpConstraintGE,
#             rhs=self.betas[j-1],
#             name="constraint_{0}".format(j)))
#             for j,label in zip(set_J, self.labels)}

#             constraints_new = {0 : problem.addConstraint(
#             LpConstraint(
#             e=lpSum(candidates[i] for i in candidates_idx),
#             sense=LpConstraintLE,
#             rhs=self.M,
#             name="Eq_M"))
#             }

#             constraints_new2 = dict()
#             if self.type == 'prob':
#                 for label in self.labels:
#                     constraints_new2[label] = {idx: problem.addConstraint(
#                     LpConstraint(
#                     e= lpSum(candidates[i]*(self.taus[label+'_80_PERC_PROB']-test_dataframe.loc[i,label+'_probs']) for i in [idx]),
#                     sense=LpConstraintLE,
#                     rhs=0.0,
#                     name="constraint1_{0}_{1}".format(idx, label)))
#                     for idx in candidates_idx
#                     }

#             else:
#                 for label in self.labels:
#                     constraints_new2[label] = {idx: problem.addConstraint(
#                     LpConstraint(
#                     e= lpSum(candidates[i]*(self.taus[label+'_80_PERC_ENTR']-test_dataframe.loc[i,label+'_entr']) for i in [idx] for label in self.labels),
#                     sense=LpConstraintGE,
#                     rhs=0.0,
#                     name="constraint1_{0}_{1}".format(idx, label)))
#                     for idx in candidates_idx
#                     }
#         #pdb.set_trace()
#         return problem

#     def solve_knapsack(self,problem,timelimit=5.0):
#         #solver = GUROBI_CMD(timeLimit=timelimit)
#         solver = GLPK_CMD()
#         #pdb.set_trace()
#         problem.solve()
#         try:
#             solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables() ],[v.varValue for v in problem.variables() ]))
#         except:
#             solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables()[1:]],[v.varValue for v in problem.variables()[1:]]))
#         return solution
    
#     def get_few_shot_exemplars(self,test_idx,similarity_matrix,test_dataframe,num_it=1):
#         #candidates = self.get_KNN(test_idx,similarity_matrix,train_filter)   #Can do filtering if have lots of candidates
#         if num_it == 1:
#             candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
#             #pdb.set_trace()
#             problem = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix)
#             solution = self.solve_knapsack(problem)
            
#             selection = [k for k,v in solution.items() if v==1]
#             few_shot_selection = [idx for idx in np.argsort(similarity_matrix[test_idx]) if idx in selection][::-1]
#             curr_len = len(few_shot_selection)
#             #pdb.set_trace()
#             if curr_len < self.M:
#                 for _ in range(self.M-curr_len):
#                     for idx in np.argsort(similarity_matrix[test_idx])[::-1]:
#                         if idx not in few_shot_selection and idx != test_idx:
#                             few_shot_selection.append(idx)
#                             break
#             assert len(few_shot_selection) == self.M
#             #pdb.set_trace()
#         else:
#             pdb.set_trace()
#             assert self.M == 1
#             few_shot_selection = []
#             for j in range(num_it):
#                 if j == 0:
#                     candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
#                 else:
#                     candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx and j not in few_shot_selection]
#                 problem = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix)
#                 solution = self.solve_knapsack(problem)
        
#                 selection = [k for k,v in solution.items() if v==1]
#                 #pdb.set_trace()
#                 #assert len(selection) == 1
#                 few_shot_selection.append(selection[0])

#         return few_shot_selection

class SEER_GURB:
    '''
    The SEER algorithm.
    Attributes:
        k (int) : the number of nearest neighbor to filter
        M (int) : the maximum number of exemplars to select
        alpha (float) : the share of exemplars that should possess the attribute of the test instance
        beta (float) : the share of exemplars that should not possess the attribute of the test instance
        modules (list) : list of constraint modules to include
    '''
    def __init__(self,
                 k=100,
                 M=8,
                 taus={},
                 num_it=1,
                 betas=[2, 2, 2, 2, 0],
                 labels=['PER', 'LOC', "ORG", "DATE", "O"],
                 perc='80',
                 type='prob'):  
        self.k = k
        self.M = M
        #self.alpha = alpha
        assert len(betas) == 3 and len(labels) == 3
        self.betas = betas
        self.labels = labels
        self.taus = taus
        self.num_it = num_it
        self.type = type
        self.perc = perc

    def get_KNN(self,test_idx,similarity_matrix,train_filter=[]):
        '''
        Retrieves the nearest neighbor of a test instance given a similarity matrix.
        '''
        instance_similarity = similarity_matrix[test_idx]
        candidates = [idx for idx in np.argsort(instance_similarity) if idx in train_filter][-self.k:][::-1]  #First is the best one 
        return candidates
    
    def get_knapsack(self,
                     test_idx,
                     candidates_idx,
                     test_dataframe,
                     similarity_matrix,
                     rem_lab_cons=False,
                     rem_conf_cons=False,
                     name='seer knapsack problem'):
        #assert test_idx not in candidates_idx and len(set(candidates_idx)) == 98
        problem = grb.Model(name)
        candidates = problem.addVars(candidates_idx,vtype=grb.GRB.BINARY, name="instance")
        labels_idx = list(range(len(self.labels)))

        if not rem_lab_cons:
            eps_label = problem.addVars(labels_idx,vtype=grb.GRB.CONTINUOUS, name="eps_label")
            problem.addConstrs((eps_label[i] >= 0 for i in labels_idx), name="non_neg_eps_label")
            objective = grb.quicksum(similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx) - grb.quicksum(eps_label[j] for j in labels_idx)


            # #Conf as constraint (to be removed)
            # eps_conf = problem.addVars(labels_idx,vtype=grb.GRB.CONTINUOUS, name="eps_conf")
            # problem.addConstrs((eps_conf[i] >= 0 for i in labels_idx), name="non_neg_eps_conf")
            # objective -= grb.quicksum(eps_conf[i] for i in labels_idx)

            # #Add label in primary obj.
            # for j,label in enumerate(self.labels):
            #     objective += grb.quicksum(test_dataframe.loc[i, label+"_cnt"]*candidates[i] for i in candidates_idx) - self.betas[j] + eps_label[j]
        else:
            print("Remove label cons")
            objective = grb.quicksum(similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx)


        #Add conf in primary
        if not rem_conf_cons:
            #db.set_trace()
            lamda = 100.0
            for label in self.labels:
                #pdb.set_trace()
                objective += lamda*grb.quicksum(min((test_dataframe.loc[i,label+'_probs']-self.taus[label+'_'+self.perc+'_PERC_PROB']),0)*candidates[i]*indic_pos(test_dataframe.loc[i,label+'_probs']) for i in candidates_idx)  #Added conf. condition in primary

        problem.setObjective(objective, sense=grb.GRB.MAXIMIZE)
        #Add capacity constraints
        #if self.num_it == 1:
        # for beta_,label in zip(self.betas, self.labels):
        #     problem += lpSum([test_dataframe.loc[i, label]*candidates[i] for i in candidates_idx]) > self.M * beta_   
        
        constraints_new = {0 : problem.addConstr(
        lhs=grb.quicksum(candidates[i] for i in candidates_idx),
        sense=grb.GRB.EQUAL,
        #sense=grb.GRB.LESS_EQUAL,
        rhs=self.M,
        name="Eq_M")
        }

        if not rem_lab_cons:
            set_J = range(1, len(self.labels)+1)
            constraints = {j : problem.addConstr(
            lhs=grb.quicksum(test_dataframe.loc[i, label+"_cnt"]*candidates[i] for i in candidates_idx),
            sense=grb.GRB.GREATER_EQUAL,
            #rhs=self.M*self.betas[j-1] - eps_label[j-1],
            rhs=1.0 - eps_label[j-1],
            #rhs=self.betas[j-1],
            name="constraint_{0}".format(j))
            for j,label in zip(set_J, self.labels)}


        # #Conf as a constraint
        # constraints_new2 = dict()
        # if self.type == 'prob':
        #     for j,label in enumerate(self.labels):
        #         constraints_new2[label] = {k+1+len(self.labels): problem.addConstr(
        #         lhs= grb.quicksum(candidates[i]*(self.taus[label+'_80_PERC_PROB']-test_dataframe.loc[i,label+'_probs']) for i in [idx]),
        #         sense=grb.GRB.LESS_EQUAL,
        #         rhs=eps_conf[j],
        #         name="constraint1_{0}_{1}".format(idx, label))
        #         for k,idx in enumerate(candidates_idx)
        #         }

        # else:
        #     pdb.set_trace()
        #     for j,label in enumerate(self.labels):
        #         constraints_new2[label] = {idx: problem.addConstr(
        #         lhs= grb.quicksum(candidates[i]*(self.taus[label+'_80_PERC_ENTR']-test_dataframe.loc[i,label+'_entr']) for i in [idx] for label in self.labels),
        #         sense=grb.GRB.GREATER_EQUAL,
        #         rhs=-eps_conf[j],
        #         name="constraint1_{0}_{1}".format(idx, label))
        #         for idx in candidates_idx
        #         }
    #pdb.set_trace()
        return problem, candidates

    def solve_knapsack(self,model,timelimit=5.0):
        #solver = GUROBI_CMD(timeLimit=timelimit)
        solver = GLPK_CMD()
        #pdb.set_trace()
        model.optimize()
        #pdb.set_trace()
        all_vars = model.getVars()
        #pdb.set_trace()
        solution = dict(zip([int(re.search(r'\[(\d+)\]', v.VarName).group(1)) for v in all_vars if 'instance' in v.VarName], [v.X for v in all_vars if 'instance' in v.VarName]))

        # Convert the result to a dictionary
        #solution = dict(zip(candidates.keys(), variable_values.values()))

        # Print the result
        #pdb.set_trace()
        # try:
        #     solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables() ],[v.varValue for v in problem.variables() ]))
        # except:
        #     solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables()[1:]],[v.varValue for v in problem.variables()[1:]]))
        return solution
    
    def get_few_shot_exemplars(self,test_idx,similarity_matrix,test_dataframe, rem_lab_cons, rem_conf_cons, num_it=1):
        #candidates = self.get_KNN(test_idx,similarity_matrix,train_filter)   #Can do filtering if have lots of candidates
        if num_it == 1:
            candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
            #pdb.set_trace()
            problem, _ = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix, rem_lab_cons, rem_conf_cons)
            solution = self.solve_knapsack(problem)
            
            selection = [k for k,v in solution.items() if v==1]
            few_shot_selection = [idx for idx in np.argsort(similarity_matrix[test_idx]) if idx in selection][::-1]
            curr_len = len(few_shot_selection)
            #pdb.set_trace()
            assert curr_len == self.M
            for _ in range(self.M-curr_len):
                for idx in np.argsort(similarity_matrix[test_idx])[::-1]:
                    if idx not in few_shot_selection and idx != test_idx:
                        few_shot_selection.append(idx)
                        break
            assert len(few_shot_selection) == self.M
            #pdb.set_trace()
        else:
            pdb.set_trace()
            assert self.M == 1
            few_shot_selection = []
            for j in range(num_it):
                if j == 0:
                    candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
                else:
                    candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx and j not in few_shot_selection]
                problem = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix)
                solution = self.solve_knapsack(problem)
        
                selection = [k for k,v in solution.items() if v==1]
                #pdb.set_trace()
                #assert len(selection) == 1
                few_shot_selection.append(selection[0])

        return few_shot_selection, curr_len



class SEER_SEQ_GURB:
    '''
    The SEER algorithm.
    Attributes:
        k (int) : the number of nearest neighbor to filter
        M (int) : the maximum number of exemplars to select
        alpha (float) : the share of exemplars that should possess the attribute of the test instance
        beta (float) : the share of exemplars that should not possess the attribute of the test instance
        modules (list) : list of constraint modules to include
    '''
    def __init__(self,
                 k=100,
                 M=8,
                 taus={},
                 num_it=1,
                 betas=[2, 2, 2, 0],
                 labels=['PER', 'LOC', "ORG", "O"],
                 type='prob'):  
        self.k = k
        self.M = M
        #self.alpha = alpha
        self.betas = betas
        self.labels = labels
        self.taus = taus
        self.num_it = num_it
        self.type = type

    def get_KNN(self,test_idx,similarity_matrix,train_filter=[]):
        '''
        Retrieves the nearest neighbor of a test instance given a similarity matrix.
        '''
        instance_similarity = similarity_matrix[test_idx]
        candidates = [idx for idx in np.argsort(instance_similarity) if idx in train_filter][-self.k:][::-1]  #First is the best one 
        return candidates
    
    def get_knapsack(self,
                     test_idx,
                     candidates_idx,
                     test_dataframe,
                     similarity_matrix,
                     rem_lab_cons=False,
                     rem_conf_cons=False,
                     name='seer knapsack problem'):
        #assert test_idx not in candidates_idx and len(set(candidates_idx)) == 99
        problem = grb.Model(name)
        candidates = problem.addVars(candidates_idx,vtype=grb.GRB.BINARY, name="instance")
        labels_idx = list(range(len(self.labels)))

        if not rem_lab_cons:
            eps_label = problem.addVars(labels_idx,vtype=grb.GRB.CONTINUOUS, name="eps_label")
            #eps_conf = problem.addVars(labels_idx,vtype=grb.GRB.CONTINUOUS, name="eps_conf")
            problem.addConstrs((eps_label[i] >= 0 for i in labels_idx), name="non_neg_eps_label")
            #problem.addConstrs((eps_conf[i] >= 0 for i in labels_idx), name="non_neg_conf_label")
            objective = grb.quicksum(similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx) - grb.quicksum(eps_label[j] for j in labels_idx)


            # #Conf as constraint (to be removed)
            # eps_conf = problem.addVars(labels_idx,vtype=grb.GRB.CONTINUOUS, name="eps_conf")
            # problem.addConstrs((eps_conf[i] >= 0 for i in labels_idx), name="non_neg_eps_conf")
            # objective -= grb.quicksum(eps_conf[i] for i in labels_idx)

            # #Add label in primary obj.
            # for j,label in enumerate(self.labels):
            #     objective += grb.quicksum(test_dataframe.loc[i, label+"_cnt"]*candidates[i] for i in candidates_idx) - self.betas[j] + eps_label[j]
        else:
            #pdb.set_trace()
            print("Remove label cons")
            objective = grb.quicksum(similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx)


        #Add conf in primary
        if not rem_conf_cons:
            for k,label in enumerate(self.labels):
                #all_lamdas = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
                lamda = 1.0
                #pdb.set_trace()
                #objective += lamda*grb.quicksum((test_dataframe.loc[i,label+'_probs']-self.taus[label+'_80_PERC_PROB'])*candidates[i]*indic_pos(test_dataframe.loc[i,label+'_probs']) for i in candidates_idx)  #v1
                #objective += lamda*grb.quicksum(min((test_dataframe.loc[i,label+'_probs']-self.taus[label+'_80_PERC_PROB']),0)*candidates[i]*indic_pos(test_dataframe.loc[i,label+'_probs']) for i in candidates_idx)  #min_v2
                #objective += lamda*grb.quicksum((test_dataframe.loc[i,label+'_probs'])*candidates[i]*indic_pos(test_dataframe.loc[i,label+'_probs']) for i in candidates_idx)  #no tau v3
                #objective += lamda*grb.quicksum((test_dataframe.loc[i,label+'_probs'] - self.taus[label+'_80_PERC_PROB'])*candidates[i] for i in candidates_idx)  #v4 (original)
                try:
                    objective += lamda*grb.quicksum(min((test_dataframe.loc[i,label+'_probs']-self.taus[label+'_80_PERC_PROB']),0)*candidates[i] for i in candidates_idx)  #v5
                except:
                    pdb.set_trace()
                #objective += lamda*grb.quicksum(min((test_dataframe.loc[i,label+'_probs']-1.0),0)*candidates[i] for i in candidates_idx)  #v6
                #objective += lamda*grb.quicksum(min((test_dataframe.loc[i,label+'_probs']),0)*candidates[i] for i in candidates_idx)  #v7
                #pdb.set_trace()
                #objective += lamda*grb.quicksum(test_dataframe.loc[i,label+'_probs']*candidates[i] for i in candidates_idx)   # V6
        else:
            #pdb.set_trace()
            pass
        problem.setObjective(objective, sense=grb.GRB.MAXIMIZE)
        #Add capacity constraints
        #if self.num_it == 1:
        # for beta_,label in zip(self.betas, self.labels):
        #     problem += lpSum([test_dataframe.loc[i, label]*candidates[i] for i in candidates_idx]) > self.M * beta_   
        
        constraints = {0 : problem.addConstr(
        lhs=grb.quicksum(candidates[i] for i in candidates_idx),
        sense=grb.GRB.EQUAL,
        #sense=grb.GRB.LESS_EQUAL,
        rhs=self.M,
        name="Eq_M")
        }
        #pdb.set_trace()
        if not rem_lab_cons:
            set_J = range(1, len(self.labels)+1)
            constraints = {j : problem.addConstr(
            lhs=grb.quicksum(test_dataframe.loc[i, label+"_cnt"]*candidates[i] for i in candidates_idx),
            sense=grb.GRB.GREATER_EQUAL,
            rhs=self.M*self.betas[j-1] - eps_label[j-1],
            #rhs=self.betas[j-1],
            name="constraint_{0}".format(j))
            for j,label in zip(set_J, self.labels)}


        # #Conf as a constraint
        # constraints_new2 = dict()
        # if self.type == 'prob':
        #     for j,label in enumerate(self.labels):
        #         constraints_new2[label] = {k+1+len(self.labels): problem.addConstr(
        #         lhs= grb.quicksum(candidates[i]*(self.taus[label+'_80_PERC_PROB']-test_dataframe.loc[i,label+'_probs']) for i in [idx]),
        #         sense=grb.GRB.LESS_EQUAL,
        #         rhs=eps_conf[j],
        #         name="constraint1_{0}_{1}".format(idx, label))
        #         for k,idx in enumerate(candidates_idx)
        #         }

        # else:
        #     pdb.set_trace()
        #     for j,label in enumerate(self.labels):
        #         constraints_new2[label] = {idx: problem.addConstr(
        #         lhs= grb.quicksum(candidates[i]*(self.taus[label+'_80_PERC_ENTR']-test_dataframe.loc[i,label+'_entr']) for i in [idx] for label in self.labels),
        #         sense=grb.GRB.GREATER_EQUAL,
        #         rhs=-eps_conf[j],
        #         name="constraint1_{0}_{1}".format(idx, label))
        #         for idx in candidates_idx
        #         }
    #pdb.set_trace()
        return problem, candidates

    def solve_knapsack(self,model,timelimit=5.0):
        #solver = GUROBI_CMD(timeLimit=timelimit)
        solver = GLPK_CMD()
        #pdb.set_trace()
        model.optimize()
        #pdb.set_trace()
        all_vars = model.getVars()
        #pdb.set_trace()
        solution = dict(zip([int(re.search(r'\[(\d+)\]', v.VarName).group(1)) for v in all_vars if 'instance' in v.VarName], [v.X for v in all_vars if 'instance' in v.VarName]))

        # Convert the result to a dictionary
        #solution = dict(zip(candidates.keys(), variable_values.values()))

        # Print the result
        #pdb.set_trace()
        # try:
        #     solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables() ],[v.varValue for v in problem.variables() ]))
        # except:
        #     solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables()[1:]],[v.varValue for v in problem.variables()[1:]]))
        return solution
    
    def get_few_shot_exemplars(self,test_idx,similarity_matrix,test_dataframe, rem_lab_cons, rem_conf_cons, num_it=1, cand_size=None):
        #candidates = self.get_KNN(test_idx,similarity_matrix,train_filter)   #Can do filtering if have lots of candidates
        if num_it == 1:
            candidates = [j for j in list(range(similarity_matrix.shape[0]))[:cand_size] if j != test_idx]
            #pdb.set_trace()
            problem, _ = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix, rem_lab_cons, rem_conf_cons)
            solution = self.solve_knapsack(problem)
            
            selection = [k for k,v in solution.items() if v==1]
            few_shot_selection = [idx for idx in np.argsort(similarity_matrix[test_idx]) if idx in selection][::-1]
            curr_len = len(few_shot_selection)
            #pdb.set_trace()
            for _ in range(self.M-curr_len):
                for idx in np.argsort(similarity_matrix[test_idx])[::-1]:
                    if idx not in few_shot_selection and idx != test_idx:
                        few_shot_selection.append(idx)
                        break
            assert len(few_shot_selection) == self.M
            #pdb.set_trace()
        else:
            pdb.set_trace()
            assert self.M == 1
            few_shot_selection = []
            for j in range(num_it):
                if j == 0:
                    candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx]
                else:
                    candidates = [j for j in list(range(similarity_matrix.shape[0])) if j != test_idx and j not in few_shot_selection]
                problem = self.get_knapsack(test_idx,candidates,test_dataframe,similarity_matrix)
                solution = self.solve_knapsack(problem)
        
                selection = [k for k,v in solution.items() if v==1]
                #pdb.set_trace()
                #assert len(selection) == 1
                few_shot_selection.append(selection[0])
        print(few_shot_selection)
        return few_shot_selection, curr_len
