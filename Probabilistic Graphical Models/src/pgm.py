'''
Created on Feb 9, 2013

@author: Justin
'''

import scipy
try:
    from collections.abc import Sequence as AbstractSequence
except ImportError:
    from collections import Sequence as AbstractSequence
from pygraph.classes.graph import graph as Graph
from pygraph.classes.digraph import digraph as DirectedGraph
from pygraph.algorithms.cycles import find_cycle

#DONE: Implement 'RandomVariable' class
#DONE: Implement 'Factor' class
#TODO: Implement 'Model' class
#DONE: Assignment #1: Introduction to Bayesian Networks
#TODO: Assignment #2: Bayes Nets for Genetic Inheritance
#TODO: Assignment #3: Markov Networks for OCR
#TODO: Assignment #4: Exact Inference
#TODO: Assignment #5: Approximate Inference
#TODO: Assignment #6: Decision Making
#TODO: Assignment #7: CRF Learning for OCR
#TODO: Assignment #8: Learning Tree-structured Networks
#TODO: Assignment #9: Learning with Incomplete Data

class Subject:
    """
    Used as a base class for the Subject of the Observer Pattern.
    """
    def __init__(self, *args, **kwargs):
        self._observers = set()
    def _observe(self, *args, **kwargs):
        for observer in self._observers:
            observer.notify(self, *args, **kwargs)
    def register(self, observer):
        self._observers.add(observer)
    def unregister(self, observer):
        self._observers.discard(observer)
        
    
class Observer:
    """
    Used as a base class for the Observer of the Observer Pattern.
    """
    def __init__(self, *args, **kwargs):
        pass
    def _register(self, subject):
        subject.register(self)
    def _unregister(self, subject):
        subject.unregister(self)
    def notify(self, subject, *args, **kwargs):
        raise NotImplementedError
        
    
class RandomVariable(Subject, AbstractSequence):
    """
    Each RandomVariable is a collection of distinct values
    representing the allowed instantiations of that random variable,
    commonly called the sample space or state space.
    
    A RandomVariable can be "observed" (in the scientific sense)
    in one of its states. If so, then it will notify all observers that
    its state has been set, but they must query (at their discretion)
    to know what state that is.
    
    A RandomVariable which is not "observed" is called "hidden", or, rarely,
    "mixed".
    
    The RandomVariable is also responsible for knowing the size of its
    state space, i.e. its cardinality.
    
    Examples:
    (1) Binary random variable has state space (0, 1),
        a.k.a. coin-flip ('H', 'T')
    (2) The "classic" FICO score has a state space (300, 301, ..., 850)
        http://en.wikipedia.org/wiki/Credit_score_in_the_United_States#FICO_score_range
    
    In Physics, a RandomVariable is typically called a Local Field (Variable)
    or, in other contexts, a (Quantum) State.
    
    ---FOR NOW: ONLY IMPLEMENT DISCRETE RANDOM VARIABLES---
    TODO: Implement continuous random variables
    """
    
    _hidden_state = None
    
    def _broadcasts(self = None):
        #TODO: Think about this design choice -- is this really what I want?
        def decorator(func):
            def wrapper(self, *__args, **__kwargs):
                wrapper.__name__ = func.__name__
                wrapper.__doc__ = func.__doc__
                func(self, *__args, **__kwargs)
                self._observe()
            return wrapper
        return decorator
    
    def __init__(self, states):
        """
        Construct a RandomVariable with the chosen state space,
        representing the possible realizations of some random variable.
        
        states -- the allowed (unique) instantiations of this random variable
        """
        super().__init__()
        assert isinstance(states, AbstractSequence), \
                "The states must form an indexed collection."
        self._ordering = {state : index for (index, state) in enumerate(states)}
        if self._hidden_state in self._ordering:
            raise KeyError("{} is not a valid state.".format(self._hidden_state))
        #the following will reconstruct the states in the original order,
        #but as a tuple and, furthermore, with duplicates removed
        self._states = tuple(sorted(self._ordering.keys(), key = self._ordering.get))
        self._cardinality = len(self._states)
        self._set_hidden_state()
        
    #AbstractSequence methods
    def __getitem__(self, index):
        return self._states[index]
    def __len__(self):
        return self._cardinality
    def __contains__(self, value):
        return value in self._ordering
    def __iter__(self):
        return iter(self._states)
    def __reversed__(self):
        return reversed(self._states)
    def index(self, value = None):
        if value is None:
            value = self._state
        return self._ordering.get(value)
    def count(self, value):
        #only returns 0 or 1 since the states are unique
        return 1 if value in self._ordering else 0
    
    def _set_hidden_state(self):
        self._hidden = True
        self._state = self._hidden_state
        
    @property
    def states(self):
        return self._states
    @property
    def cardinality(self):
        return self._cardinality
    @property
    def hidden(self):
        return self._hidden
    @property
    def state(self):
        return self._state
    @state.setter
    @_broadcasts()
    def state(self, s):
        assert s in self._ordering, \
            "Invalid state assignment, {}, selected.".format(s)
        self._hidden = False
        self._state = s
    
    @_broadcasts()
    def reset(self):
        self._set_hidden_state()

class Factor(Observer):
    """
    A Factor is a mapping between a collection of state spaces
    (conceptually, instantiations of random variables within those spaces)
    to some subset of the real numbers, called "beliefs".
    
    This definition includes, as practical examples:
    (1) discrete factors (factors over a discrete, finite domain)
    (2) probabilities (normalized discrete factors)
    (3) continuous factors (factors over continuous or infinite domain)
    (4) probability density functions (normalized continuous factors)
    (5) log-probabilities (wherein the range of the factor map is
                    further mapped to log-space to prevent underflow)
                    
    In Physics, a Factor is typically called a (Field) Potential or
    a Local (Field) Operator.
    
    ---FOR NOW: ONLY IMPLEMENT (1 & 2)---
    TODO: Implement (3) through (5)
    """
    
    #TODO: Design question... Should these all be static methods?"   
    @staticmethod
    def marginalize(factor, scope):
        """
        Return a Factor that has the beliefs of the input factor
        about the random variables in the input scope "summed out".
        
        The scope of the output will match that of the input factor,
        except the random variables of the input scope will be removed.
        
        If this results in an empty scope, then a Factor will be created
        that has a 0-rank array for its beliefs. This is useful when
        calculating the partition function, which can be thought of as
        the joint distribution marginalized to an empty scope.
        
        Any random variables in the input scope that are not in the
        input factor's scope will be silently ignored.
        
        Does not modify the input.
        """
        removal_scope = list(set(factor.scope) & set(scope))
        restricted_scope = list(set(factor.scope) - set(removal_scope))
        summed_axes = list(map(factor.scope.index, removal_scope))
        #apply_over_axes produces an array of the same shape as factor.beliefs;
        #however, we want the reduced dimensionality matching the restricted scope,
        #otherwise we will violate the constraints on Factor construction...
        #Warning: this may produce a 0-rank array
        summed_beliefs = scipy.apply_over_axes(scipy.sum, factor.beliefs,
                                               summed_axes).squeeze()
        return Factor(restricted_scope, summed_beliefs)
    @staticmethod
    def product(factorA, factorB):
        """
        Return a Factor that combines the scope and beliefs of the input factors.
        
        Does not modify the input.
        
        factorA.scope, factorB.scope -> new_factor.scope:
        [c, a, d, e], [e, f, d] -> [c, a, d, e, f]
        factorA.beliefs.shape, factorB.beliefs.shape -> new_factor.shape:
        (4, 2, 5, 6), (6, 7, 5) -> (4, 2, 5, 6, 7)
        """
        B_not_A_scope = [rvar for rvar in factorB.scope if rvar not in factorA.scope]
        combined_scope = factorA.scope + B_not_A_scope
        subscopeB = [rvar for rvar in combined_scope if rvar in factorB.scope]
        permutationB = tuple(map(factorB.scope.index, subscopeB))
        sliceA = tuple(slice(None) if rvar in factorA.scope else scipy.newaxis
                       for rvar in combined_scope)
        sliceB = tuple(slice(None) if rvar in factorB.scope else scipy.newaxis
                       for rvar in combined_scope)
        #the variables in factorA.scope remain un-permuted in the combined scope
        beliefA = factorA.beliefs[sliceA]
        beliefB = factorB.beliefs.transpose(permutationB)[sliceB]
        combined_beliefs = beliefA * beliefB
        return Factor(combined_scope, combined_beliefs)
    @staticmethod
    def joint(*factors):
        """
        Return a Factor that combines the scope and beliefs of the input factors.
        
        Does not modify the input.
        
        factorA.scope, factorB.scope, factorC.scope
            -> new_factor.scope:
        [c, a, d, e], [e, f, d], [g, e, a]
            -> [c, a, d, e, f, g] up to permutation
        factorA.beliefs.shape, factorB.beliefs.shape, factorC.beliefs.shape
            -> new_factor.shape:
        (4, 2, 5, 6), (6, 7, 5), (3, 6, 2)
            -> (4, 2, 5, 6, 7, 3) up to permutation
        """
        joint_scope = list(set.union(*(set(factor.scope) for factor in factors)))
        subscopes = [[rvar for rvar in joint_scope if rvar in factor.scope]
                        for factor in factors]
        permutations = [tuple(map(factor.scope.index, subscope))
                        for (factor, subscope) in zip(factors, subscopes)]
        slices = [tuple(slice(None) if rvar in factor.scope else scipy.newaxis
                        for rvar in joint_scope) for factor in factors]
        beliefs = [factor.beliefs.transpose(permutation)[slice_]
                   for (factor, permutation, slice_) in zip(factors, permutations, slices)]
        joint_beliefs = scipy.multiply.reduce(beliefs)
        return Factor(joint_scope, joint_beliefs)
    @staticmethod
    def reordered(factor, target_scope):
        """
        Return a Factor that has the target scope and, up to reordering,
        matches the beliefs of the input factor.
        
        factor.scope, target_scope -> new_factor.scope:
        [a, b, c, d], [c, d, b, a] -> [c, d, b, a]
        factor.beliefs.shape, target_scope -> new_factor.beliefs.shape:
        (2, 3, 4, 5), [c, d, b, a] -> (4, 5, 3, 2)
        """
        #TODO: Strengthen this check or validate some other way."
        assert set(factor.scope) == set(target_scope), \
            "The factor scope and target scope must be related by a permutation."
        permutation = tuple(map(factor.scope.index, target_scope))
        permuted_scope = [factor.scope[j] for j in permutation]
        return Factor(permuted_scope, factor.beliefs.transpose(permutation))
    
    def _mutates_proxies(self = None):
        #TODO: Think about this design choice -- is this really what I want?
        def decorator(func):
            def wrapper(self, *__args, **__kwargs):
                wrapper.__name__ = func.__name__
                wrapper.__doc__ = func.__doc__
                func(self, *__args, **__kwargs)
                self._proxify()
            return wrapper
        return decorator
        
    @_mutates_proxies()
    def __init__(self, scope, beliefs):
        """
        Construct a Factor that knows about the random variables in the input
        scope and has the input beliefs about them.
        
        scope   -- an indexed collection of distinct random variables that
                    this factor has beliefs about
        beliefs -- a scipy/numpy array (possibly 0-rank) of real numbers with
                    (1) beliefs.ndim == len(scope)
                    (2) beliefs.shape[k] == scope[k].cardinality for all k
                    (3) Order of axes should reflect the order of random variables
                        in the scope!    
        """
        super().__init__()
        if not isinstance(beliefs, scipy.ndarray):
            raise NotImplementedError("'beliefs' must be a scipy/numpy array.")
        assert all(isinstance(belief, (scipy.integer, scipy.floating))
                   for belief in beliefs.flat), \
                "The beliefs must all be real numbers."
        assert beliefs.ndim == len(scope), \
                "The beliefs do not match the scope."
        assert isinstance(scope, AbstractSequence), \
                "The scope must be an indexed collection."
        assert all(isinstance(rvar, RandomVariable) for rvar in scope), \
                "The scope must be a collection of random variables."
        assert all(num_states == rvar.cardinality
                   for (num_states, rvar) in zip(beliefs.shape, scope)), \
                "The beliefs do not match the cardinalities of the random variables."
        self._scope = scope
        self._beliefs = beliefs
        for rvar in self._scope:
            self._register(rvar)
        
    def _proxify(self):
        self._proxy_scope = [rvar for rvar in self._scope if rvar.hidden]
        subslice = tuple(slice(None) if rvar.hidden else rvar.index()
                         for rvar in self._scope)
        #makes subtle use of [()] indexing in case belief is 0-rank
        #self._beliefs is -always- an ndarray, whereas
        #self._proxy_beliefs is often an ndarray, but may be a scalar,
        #typically when self._proxy_scope is empty
        self._proxy_beliefs = self._beliefs[subslice]
        
    @_mutates_proxies()
    def notify(self, rvar):
        pass
        
    @property
    def scope(self):
        return self._proxy_scope
    @property
    def beliefs(self):
        return self._proxy_beliefs
    @property
    def probabilities(self):
        return self.beliefs / self.beliefs.sum()
    
    def sample(self):
        cumulative_measures = self.beliefs.cumsum().reshape(self.beliefs.shape)
        random_measure = scipy.random.random() * self.beliefs.sum()
        indices = scipy.argwhere(cumulative_measures > random_measure)[0]
        return tuple(rvar[index] for (rvar, index) in zip(self.scope, indices))

class CPD(Factor):
    """
    A CPD, or "Conditional Probability Distribution", is a Factor representing
    P(child | parents).
    
    Its scope should be of the form [child, parent1, parent2, ...].
    Its beliefs will initially be normalized along the child axis. 
    """
    def __init__(self, scope, beliefs):
        self._child = scope[0]
        self._parents = scope[1 : ]
        super().__init__(self, scope, beliefs / beliefs.sum(axis = 0))
    
    @property
    def child(self):
        return self._child
    @property
    def parents(self):
        return self._parents

class BipartiteGraph(Graph):
    """
    A BipartiteGraph is a Graph in which the vertices can be divided into
    two disjoint sets such that all edges connect vertices in different subsets.
    
    G = (U, V, E) is the usual representation of such a graph,
    with e == (u, v) for each e in E, for some (u, v) pair in UxV.
    
    However, a more concise representation, used here, is of the form:
    {'u_1' : ['v_4', 'v_3'], ... 'u_m' : ['v_1', 'v_3', 'v_n']}
    This is essentially the adjacency list representation.
    """
    def __init__(self, adjacencies):
        """
        Construct a BipartiteGraph connecting vertices according to the input
        adjacencies.
        
        adjacencies -- a dictionary of adjacency lists containing node names as strings
        """
        super().__init__()
        self.add_nodes(set.union(set(adjacencies.keys()),
                                     *map(set, adjacencies.values())))
        for (u, vs) in adjacencies.items():
            for v in vs:
                self.add_edge((u, v))
    
class DirectedAcyclicGraph(DirectedGraph):
    #TODO: Figure out why I'm repeating myself here, re: BipartiteGraph
    def __init__(self, adjacencies):
        super().__init__()
        self.add_nodes(set.union(set(adjacencies.keys()),
                                     *map(set, adjacencies.values())))
        for (u, vs) in adjacencies.items():
            for v in vs:
                self.add_edge((u, v))
        if find_cycle(self):
            raise TypeError("Cycle detected!")
    
class BayesNet:
    def __init__(self, cpds, rvar_labels):
        self._cpds = cpds
        self._rvars = rvar_labels.keys()
        self._adjacencies = {rvar_labels[cpd.child] : [rvar_labels[parent]
                             for parent in cpd.parents] for cpd in cpds}
        self._graph = DirectedAcyclicGraph(self._adjacencies).reverse()
        
class Model:
    """
    A Model is a collection of RandomVariables together with Factors that encode
    beliefs about (subsets of) those variables and their interdependencies.
    
    Important examples include:
    (1) Bayesian Networks (including Bayes Classifiers, 2TBNs)
    (2) Markov Random Fields (including Ising Models, Restricted Boltzmann Machines)
    (3) Non-graphical Models (?)
    
    ---FOR NOW: ONLY IMPLEMENT (1) and (2) ---
    """
    
    def __init__(self):
        """
        Constructor
        """
        pass
    

if __name__ == '__main__':
    pass