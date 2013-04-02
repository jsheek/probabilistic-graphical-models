"""
Copyright 2013, Justin P. Sheek <jsheek@gmail.com>

This file is part of probabilistic-graphical-models, hereafter referred
to as PGM.

    PGM is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PGM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PGM.  If not, see <http://www.gnu.org/licenses/>.
"""

'''
Created on Feb 9, 2013

@author: Justin
'''

import scipy
try:
    from collections.abc import Sequence as AbstractSequence
except ImportError:
    from collections import Sequence as AbstractSequence
from pygraph.classes.graph import graph as UndirectedGraph
from pygraph.classes.digraph import digraph as DirectedGraph
from pygraph.algorithms.sorting import topological_sorting
from pygraph.algorithms.searching import breadth_first_search
from pygraph.algorithms.searching import depth_first_search
from pygraph.algorithms.pagerank import pagerank
from pygraph.algorithms.minmax import minimal_spanning_tree
from pygraph.algorithms.cycles import find_cycle

class Subject:
    """
    Used as a base class for the Subject of the Observer Pattern.
    """
    def __init__(self):
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
    def __init__(self):
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
        a.k.a. the coin-flip ('H', 'T')
    (2) The "classic" FICO score has a state space (300, 301, ..., 850)
        http://en.wikipedia.org/wiki/Credit_score_in_the_United_States#FICO_score_range
    
    In Physics, a RandomVariable is typically called a Local Field (Variable)
    or, in other contexts, a (Quantum) State.
    
    The definition here differs from the usual definition of a random variable
    in -at least- two ways:
    (i) The usual definition often requires a state space consisting of real-
        valued states. In contrast, this object is more like a "random element".
    (ii) More importantly, the usual definition often neglects to differentiate
        between random variables and beliefs -about- those variables
        (e.g. probability assignments). For more information, see:
        http://en.wikipedia.org/wiki/Mind_projection_fallacy
    
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
        Return a RandomVariable with the chosen state space,
        representing the possible realizations of this random variable.
        
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
    def divergence(factorA, factorB):
        """
        Return the divergence, or relative entropy, of the input factors.
        
        The factors must be defined over the same scope.
        
        Only makes probabilistic sense for joint factors.
        
        Z_A = E_A[1]
        Z_B = E_B[1]
        -E_A[ln(A / B)] / Z_A + ln(Z_B / Z_A)
        """
        reordered_factorB = Factor.reordered(factorB, factorA.scope)
        beliefsA = factorA.beliefs
        beliefsB = reordered_factorB.beliefs
        Z_A = beliefsA.sum()
        Z_B = beliefsB.sum()
        s = - beliefsA * scipy.log(beliefsB / beliefsA) / Z_A
        #ignore nan values resulting from 0 * log(0 / 0)
        S = s[~scipy.isnan(s)].sum()
        return S + scipy.log(Z_B / Z_A)
    @staticmethod
    def product(factorA, factorB):
        """
        Return a Factor that combines the scope and beliefs of the input factors.
        
        factorA.scope, factorB.scope -> new_factor.scope:
        [c, a, d, e], [e, f, d] -> [c, a, d, e, f]
        factorA.beliefs.shape, factorB.beliefs.shape -> new_factor.shape:
        (4, 2, 5, 6), (6, 7, 5) -> (4, 2, 5, 6, 7)
        
        Does not modify the input.
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
        
        factorA.scope, factorB.scope, factorC.scope
            -> new_factor.scope:
        [c, a, d, e], [e, f, d], [g, e, a]
            -> [c, a, d, e, f, g] up to permutation
        factorA.beliefs.shape, factorB.beliefs.shape, factorC.beliefs.shape
            -> new_factor.shape:
        (4, 2, 5, 6), (6, 7, 5), (3, 6, 2)
            -> (4, 2, 5, 6, 7, 3) up to permutation
            
        Does not modify the input.
        """
        joint_scope = list(set.union(set(), *(set(factor.scope)
                        for factor in factors)))
        #the order that the joint scope variables appear in each factor's scope
        subscopes = [[rvar for rvar in joint_scope if rvar in factor.scope]
                        for factor in factors]
        #the index mapping of those subscopes to each factor's scope
        permutations = [tuple(map(factor.scope.index, subscope))
                        for (factor, subscope) in zip(factors, subscopes)]
        #make room for new variables in order to broadcast to a common shape
        slices = [tuple(slice(None) if rvar in factor.scope
                        else scipy.newaxis for rvar in joint_scope)
                        for factor in factors]
        beliefs = [factor.beliefs.transpose(permutation)[slice_]
                        for (factor, permutation, slice_)
                        in zip(factors, permutations, slices)]
        #scipy.multiply.reduce may not return an array in all circumstances;
        #furthermore, it may return an array of sub-arrays
        #TODO: rely on numpy 1.70 for access to keepdims = True
        #joint_beliefs = scipy.multiply.reduce(beliefs, keepdims = True).squeeze()
        #FOR NOW, use broadcast_arrays to match shapes of all beliefs
        #before the multiply operation to prevent the array nesting bug...
        #also, scipy.broadcast_arrays(*[]) fails so we need an explicit check
        if not beliefs:
            return Factor.null()
        joint_beliefs = scipy.multiply.reduce(scipy.broadcast_arrays(*beliefs))
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
        
        Does not modify the input.
        """
        if set.symmetric_difference(set(factor.scope), set(target_scope)):
            raise ValueError("The factor scope and target scope must be related by a permutation.")
        permutation = tuple(map(factor.scope.index, target_scope))
        #permuted_scope = [factor.scope[j] for j in permutation]
        return Factor(target_scope, factor.beliefs.transpose(permutation))
    @staticmethod
    def uniform(target_scope):
        """
        Return a Factor whose beliefs are uniform over the target scope's states.
        """
        target_shape = tuple(rvar.cardinality for rvar in target_scope)
        uniform_beliefs = scipy.ones(target_shape)
        return Factor(target_scope, uniform_beliefs)
    @staticmethod
    def random(target_scope):
        """
        Return a randomly generated Factor that has the target scope.
        """
        target_shape = tuple(rvar.cardinality for rvar in target_scope)
        random_beliefs = scipy.random.random(target_shape)
        return Factor(target_scope, random_beliefs)
    @staticmethod
    def null():
        """
        Return a null Factor.
        """
        null_scope = []
        null_beliefs = scipy.array(scipy.NaN)
        return Factor(null_scope, null_beliefs)
    
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
        Return a Factor that knows about the random variables in the input
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
        assert issubclass(beliefs.dtype.type, (scipy.integer, scipy.floating)), \
                "The beliefs must all be real numbers."
        assert beliefs.ndim == len(scope), \
                "The beliefs do not match the scope."
        assert isinstance(scope, AbstractSequence), \
                "The scope must be an indexed collection."
        assert all(isinstance(rvar, RandomVariable) for rvar in scope), \
                "The scope must be a collection of random variables."
        assert len(set(scope)) == len(scope), \
                "The scope must contain unique random variables."
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
        #only makes probabilistic sense in the case of joint distributions
        return self.beliefs / self.beliefs.sum()
    
    def sample(self):
        #interesting idea, but, presumably, it only makes probabilistic sense
        #in the case of joint distributions
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
    #TODO: Is this the right design choice?
    #I'm not liking the consequences of how this wraps Factor.
    def __init__(self, scope, beliefs):
        self._child = scope[0]
        self._parents = scope[1 : ]
        super().__init__(scope, beliefs / beliefs.sum(axis = 0))
    
    @property
    def child(self):
        return self._child
    @property
    def parents(self):
        return self._parents

class Adjacency(dict):
    """
    Encapsulates the adjacency list representation of a graph, although
    as a dict.
    
    Reversing an Adjacency will result in a low-overhead representation
    with each edge having reversed orientation.
    
    Uses lazy evaluation whenever possible to accommodate very large graphs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heads = set()
        self._tails = set()
        self._forward = True
        
    def __iter__(self):
        return self._view(self._forward)
    def __reversed__(self):
        return self._view(not self._forward)
    def __str__(self):
        if self._forward:
            edge_symbol = '->'
        else:
            edge_symbol = '<-'
        return ''.join([edge_symbol, super().__str__()])
    
    def _view(self, forward):
        if forward:
            return self._edge_gen()
        else:
            return self._rev_edge_gen()
    def _edge_gen(self):
        for (head, tails) in self.items():
            for tail in tails:
                yield (head, tail)
    def _rev_edge_gen(self):
        return (tuple(reversed(edge)) for edge in self._edge_gen())
    
    @property
    def edges(self):
        return iter(self)
    @property
    def heads(self):
        if not self._heads:
            self._heads = set(self.keys())
        return self._heads
    @property
    def tails(self):
        if not self._tails:
            self._tails = set.union(set(), *map(set, self.values()))
        return self._tails
    @property
    def nodes(self):
        return set.union(self.heads, self.tails)
    
    def reverse(self):
        (self._heads, self._tails) = (self.tails, self.heads)
        self._forward = not self._forward
        
class AdjacencyBuilderMixin:
    """
    Adds a method that builds a graph-like object from an adjacency list
    representation of the form:
    
    {'u_1' : ['v_4', 'v_3'], ... 'u_m' : ['v_1', 'v_3', 'v_n']}
    
    Also adds convenience methods, add_nodes and add_edges, for adding several
    nodes or edges at once.
    
    Abstract mixin class, not intended for instantiation.
    """
    def __init__(self):
        pass
    
    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)
    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)
    def add_adjacencies(self, adjacencies = Adjacency()):
        """
        Add the nodes and edges represented by the adjacencies.
        """
        self.add_nodes(adjacencies.nodes)
        self.add_edges(adjacencies.edges)

class NodeScheduler:
    """
    A NodeScheduler is responsible for scheduling the visitation of nodes
    in a graph.
    
    Different algorithms and graph types will rely on different schedules.
    
    This is a useful primitive in belief propagation, Gibbs sampling, etc.
    """
    @staticmethod
    def by_weight(graph):
        raise NotImplementedError
    @staticmethod
    def round_robin(graph):
        """
        Yield nodes of the graph.
        """
        for node in graph.nodes():
            yield node
    @staticmethod
    def by_var_elim_heuristic(graph):
        """
        Yield nodes of the graph, starting from those with the fewest neighbors
        to those with the most.
        """
        for node in sorted(graph.nodes(), key = graph.node_order):
            yield node
    @staticmethod
    def level_up_down(graph):
        """
        Yield the nodes of the graph, for an arbitrary root, starting with the
        leaves, running up to the root node, and pushing back down toward the
        leaves, excepting the leaves themselves.
        
        Undefined behavior if the graph is not a tree.
        """
        arbitrary_root = next(NodeScheduler.uniform(graph))
        (_, ordering) = breadth_first_search(graph, root = arbitrary_root)
        for node in reversed(ordering):
            yield node
        for node in ordering[1 :]:
            if graph.node_order(node) > 1:
                yield node
    @staticmethod
    def topo_up_down(graph):
        """
        Yield the nodes of the graph, starting with the leaf-most
        (latest topological) node, running up to the root-most
        (earliest topological) node, and pushing back down to the leaves,
        excepting the leaf-most node.
        
        Undefined behavior if the graph is not a DAG.
        
        graph: A->B->C
        yields: (C, B, A, B)
        """
        tsort = topological_sorting(graph)
        for node in reversed(tsort):
            yield node
        for node in tsort[1 : -1]:
            yield node
    @staticmethod
    def by_pagerank(graph):
        """
        Probabilistic scheduler based on PageRank of nodes in the graph.
        
        Just for fun^^
        """
        ranks = pagerank(graph)
        rvar = RandomVariable(list(ranks.keys()))
        scope = [rvar]
        beliefs = scipy.array(list(ranks.values()))
        factor = Factor(scope, beliefs)
        while True:
            yield factor.sample()[0]
    @staticmethod
    def uniform(graph):
        """
        Yield nodes drawn uniformly at random (with replacement) from the graph.
        """
        rvar = RandomVariable(graph.nodes())
        scope = [rvar]
        beliefs = scipy.ones(len(rvar))
        factor = Factor(scope, beliefs)
        while True:
            yield factor.sample()[0]
            
    def __init__(self):
        pass
    
class TreeDecomposition:
    @staticmethod
    def virtual_var_elim(graph):
        """
        Return the TreeDecomposition associated with a virtual variable
        elimination procedure.
        
        This procedure only makes sense when the input graph is a DAG.
        """
        maximal_cliques = []
        eliminated_nodes = set()
        induced_graph = Graph.moralized(graph)
        schedule = NodeScheduler().by_var_elim_heuristic(induced_graph)
        for node in schedule:
            neighbors = set(induced_graph.neighbors(node))
            this_scope = (neighbors | {node}) - eliminated_nodes
            clique = Graph.undirected()
            clique.add_nodes(this_scope)
            clique.complete()
            induced_graph.add_graph(clique)
            if not any(scope > this_scope for scope in maximal_cliques):
                maximal_cliques.append(this_scope)
            eliminated_nodes.add(node)
        tree_dec = TreeDecomposition()
        tree_dec.graph = induced_graph
        tree_dec.cliques = maximal_cliques
        return tree_dec
    @staticmethod
    def estimate_treewidth(graph):
        """
        Almost every graph of order n has treewidth of O(n ** epsilon),
        where epsilon can be computed from the average order of each node.
        
        A corollary of this is that treewidth sucks in general.
        By extension, so does exact inference.
        """
        n = graph.order()
        node_orders = [graph.node_order(node) for node in graph.nodes()]
        delta = (sum(node_orders) + 1) / n
        epsilon = (delta - 1) / (delta + 1)
        return int(n ** epsilon)
    
    def __init__(self):
        pass
    
    @property
    def graph(self):
        return self._graph
    @property
    def cliques(self):
        return self._cliques
    @property
    def width(self):
        """
        The induced width is the size of the largest clique.
        
        (The induced width lower-bounds the running time of exact inference
        on the induced graph.)
        """
        return max(len(clique) for clique in self.cliques)
    @graph.setter
    def graph(self, graph):
        self._graph = graph
    @cliques.setter
    def cliques(self, cliques):
        self._cliques = cliques
        
class Graph:
    #TODO: work on this abstraction
    @classmethod
    def directed(cls):
        return DirectedGraph()
    @classmethod
    def undirected(cls):
        return UndirectedGraph()
    
    @staticmethod
    def clique_of(graph, node):
        neighbors = set(graph.neighbors(node))
        clique = Graph.undirected()
        clique.add_nodes(neighbors | {node})
        clique.complete()
        return clique
    @staticmethod
    def deficiency_of(graph, clique):
        return sum(1 for edge in clique.edges() if not graph.has_edge(edge)) / 2
    @staticmethod
    def moralized(graph):
        """
        Return an UndirectedGraph that has all of the input graph's nodes and
        edges, with certain edges added.
        
        The additional edges connect all parent nodes of each child node.
        
        The input graph should be a DAG so that the parent-child relationship
        is well-defined.
        """
        moral_graph = Graph.undirected()
        moral_graph.add_graph(graph)
        for node in graph.nodes():
            parent_graph = Graph.undirected()
            parent_graph.add_nodes(graph.incidents(node))
            parent_graph.complete()
            moral_graph.add_graph(parent_graph)
        return moral_graph
    @staticmethod
    def virtual_var_elim(graph, schedule = None):
        """
        Return the maximal cliques associated with a virtual variable
        elimination procedure. The cliques that are constructed will depend on
        the schedule of nodes.
        
        Finding the optimal schedule is NP-complete.
        
        This procedure only makes sense when the input graph is a DAG.
        """
        maximal_cliques = []
        induced_graph = Graph.moralized(graph)
        volatile_nodes = set(induced_graph.nodes())
        clique = {}
        deficiency = {}
        while induced_graph.nodes():
            for node in volatile_nodes:
                clique[node] = Graph.clique_of(induced_graph, node)
                deficiency[node] = Graph.deficiency_of(induced_graph, clique[node])
            node = min(deficiency.keys(), key = deficiency.get)
            this_scope = set(clique[node].nodes())
            if not any(scope > this_scope for scope in maximal_cliques):
                maximal_cliques.append(this_scope)
            markov_blankets = (set(induced_graph.neighbors(node)) for node in this_scope)
            volatile_nodes = set.union(*markov_blankets)
            induced_graph.add_graph(clique[node])
            induced_graph.del_node(node)
            volatile_nodes.discard(node)
            clique.pop(node)
            deficiency.pop(node)
        return maximal_cliques
    @staticmethod
    def weighted_by_intersection(cliques):
        weighted_clique_graph = Graph.undirected()
        weighted_clique_graph.add_nodes([tuple(clique) for clique in cliques])
        weighted_clique_graph.complete()
        for edge in weighted_clique_graph.edges():
            weight = len(set.intersection(*map(set, edge)))
            weighted_clique_graph.set_edge_weight(edge, weight)
        return weighted_clique_graph
    @staticmethod
    def minimal_spanning_tree(weighted_graph):
        """
        Return the minimal spanning tree of a weighted graph.
        
        Wraps pygraph's minimal spanning tree.
        """
        min_weight = min(weighted_graph.edge_weight(edge)
                         for edge in weighted_graph.edges())
        if min_weight < 0:
            reweighted_graph = Graph.undirected()
            reweighted_graph.add_graph(weighted_graph)
            for edge in weighted_graph.edges():
                weight = weighted_graph.edge_weight(edge)
                reweighted_graph.set_edge_weight(edge, weight - min_weight)
            return Graph.minimal_spanning_tree(reweighted_graph)
        min_tree = Graph.directed()
        min_tree.add_nodes(weighted_graph.nodes())
        for edge in minimal_spanning_tree(weighted_graph).items():
            if not None in edge:
                min_tree.add_edge(edge)
        return min_tree
    @staticmethod
    def maximal_spanning_tree(weighted_graph):
        #pygraph's minimal spanning tree algorithm does not correctly handle
        #negative edge weights, nor does it return a graph!
        #so we use this to compute the maximal spanning tree, instead of the
        #usual method of flipping signs
        min_weight = min(weighted_graph.edge_weight(edge)
                         for edge in weighted_graph.edges())
        reweighted_graph = Graph.undirected()
        reweighted_graph.add_graph(weighted_graph)
        for edge in weighted_graph.edges():
            weight = weighted_graph.edge_weight(edge)
            reweighted_graph.set_edge_weight(edge, 1 / (1 + weight - min_weight))
        return Graph.minimal_spanning_tree(reweighted_graph)
    @staticmethod
    def tree_decomposition(graph, schedule = None):
        #---recipe to turn a DAG into a clique tree---
        #(0) start with a DAG
        #(1) choose an ordering over its nodes
        #(2) use virtual variable elimination to compute maximal cliques induced by this ordering
        #(3) build a complete clique graph with edges weighted by intersection counts
        #(4) return the maximal spanning tree of that graph
        
        #since this can result in really bad widths of the resulting clique tree,
        #we will default to using "minimum deficiency search" to choose the next node on-the-fly,
        #which is the best heuristic available, it seems
        maximal_cliques = Graph.virtual_var_elim(graph, schedule)
        weighted_clique_graph = Graph.weighted_by_intersection(maximal_cliques)
        clique_tree = Graph.maximal_spanning_tree(weighted_clique_graph)
        return clique_tree
    
    def __init__(self):
        pass
    
class AdjacencyGraph(DirectedGraph, AdjacencyBuilderMixin):
    def __init__(self, adjacencies = Adjacency()):
        super().__init__()
        self.add_adjacencies(adjacencies)
    
class BipartiteGraph(UndirectedGraph, AdjacencyBuilderMixin):
    """
    A BipartiteGraph is a UndirectedGraph in which the vertices can be divided into
    two disjoint sets such that all edges connect vertices in different subsets.
    
    G = (U, V, E) is the usual representation of such a graph,
    with e == (u, v) for each e in E, for some (u, v) pair in UxV.
    
    However this relies on a more concise representation, the adjacency list.
    """
    #TODO: This doesn't maintain its own invariant... Not very useful.
    def __init__(self, adjacencies = Adjacency()):
        """
        Return a BipartiteGraph connecting vertices according to the input
        adjacencies.
        """
        super().__init__()
        U = set(adjacencies.heads)
        V = set(adjacencies.tails)
        if U & V:
            raise ValueError("Adjacencies do not form a bipartition!")
        self.add_adjacencies(adjacencies)
    
class DirectedAcyclicGraph(DirectedGraph, AdjacencyBuilderMixin):
    """
    A DirectedAcyclicGraph is a DirectedGraph which has no cycles.
    """
    #TODO: This doesn't maintain its own invariant... Not very useful.
    def __init__(self, adjacencies = Adjacency()):
        super().__init__()
        self.add_adjacencies(adjacencies)
        if find_cycle(self):
            raise ValueError("Cycle detected!")
            
class Cluster:
    """
    A Cluster is responsible for receiving messages from and emitting messages to
    a Model. The messages are generic "potentials", but primarily these are just
    joint Factors.
    
    This is a useful primitive in the belief propagation algorithm.
    """
    def __init__(self, factors):
        self._potential = Factor.joint(*factors)
        self._messages = {}
    
    @property
    def scope(self):
        return self._potential.scope
    @property
    def messages(self):
        return self._messages
    def flush(self):
        self._messages.clear()
    def receive(self, sender, message):
        unrecognized = set(message.scope) - set(self.scope)
        self._messages[sender] = Factor.marginalize(message, unrecognized)
    def emit(self, excluded = set()):
        messages = {message for (sender, message) in self._messages.items()
                    if sender not in excluded}
        return Factor.joint(self._potential, *messages)
    
class Model:
    """
    A Model is a collection of named entities.
    
    Entities can be attached.
    
    The model can be validated.
    
    TODO: Clarify this!
    """
    def __init__(self):
        """
        Constructor
        """
        self._registry = {}
    
    @property
    def registry(self):
        return self._registry
    
    def is_valid(self):
        return True
    def attach(self, registry, **kwargs):
        self._registry.update(registry, **kwargs)
    
class GraphicalModel(Model):
    """
    A GraphicalModel is a Model with an associated Graph.
    """
    def __init__(self, graph):
        super().__init__()
        self._graph = graph
    
    @property
    def graph(self):
        return self._graph
    
    def is_valid(self):
        if not super().is_valid():
            return False
        graph_ids = set(self.graph.nodes())
        registry_ids = set(self.registry.keys())
        return not set.symmetric_difference(graph_ids, registry_ids)
    def add_graph(self, graph):
        self._graph.add_graph(graph)
        
class FactorModel(Model):
    """
    A FactorModel is a collection of RandomVariables together with Factors that
    encode beliefs about (subsets of) those variables and their interdependencies.
    
    Important examples include:
    (1) Bayesian Networks (including Bayes Classifiers, 2TBNs)
    (2) Markov Random Fields (including Ising Models, Restricted Boltzmann Machines)
    (3) Factor Graphs
    (4) Non-graphical Models (?)
    
    ---FOR NOW: ONLY IMPLEMENT (1) and (2) ---
    """
    
    def __init__(self):
        super().__init__()
    
class ClusterModel(Model):
    """
    A ClusterModel is a Model that has an undirected graph representing
    the relationship between Cluster nodes.
    
    TODO: Improve this description. Explain responsibility.
    TODO: Fix __init__ and loopy_bp
    """
        
    def __init__(self, named_clusters, adjacencies):
        super().__init__()
        self.attach(named_clusters)
        self._graph = AdjacencyGraph(adjacencies)
        self._scheduler = NodeScheduler()
    
    @property
    def clusters(self):
        return self._registry.values()
    @property
    def graph(self):
        return self._graph
    @property
    def schedule(self):
        return self._scheduler.topo_up_down(self.graph)
    
    def clusters_with(self, rvar):
        return [cluster for cluster in self.clusters if rvar in cluster.scope]
    def bp(self):
        """
        Belief propagation.
        """
        for node in self.schedule:
            sender = self.registry[node]
            for neighbor in self.graph.neighbors(node):
                receiver = self.registry[neighbor]
                message = sender.emit(excluded = {neighbor})
                receiver.receive(node, message)
    def loopy_bp(self, tolerance = 1e-1, max_iter = 10):
        """
        Belief propagation for models with directed graphs that may have cycles.
        
        Approximate.
        """
        step = 0
        close = 0
        #TODO: refactor this
        n = self.graph.order()
        stats = {}
        schedule = self._scheduler.uniform(self.graph)
        for node in schedule:
            sender = self.registry[node]
            for neighbor in self.graph.neighbors(node):
                receiver = self.registry[neighbor]
                message = sender.emit(excluded = {neighbor})
                prev_message = receiver.messages.get(node, Factor.null())
                receiver.receive(node, message)
                new_message = receiver.messages.get(node, Factor.null())
                if scipy.allclose(prev_message.beliefs, new_message.beliefs, tolerance):
                    close += 1
                else:
                    close /= 2
                stats.setdefault((sender, receiver), 0)
                stats[(sender, receiver)] += 1
            step += 1
            if step > n * max_iter:
                print("Max iterations reached.")
                print(close)
                break
            if close > n:
                print("Message convergence reached.")
                print(step)
                break
        return stats
    def marginal(self, rvar):
        """
        Return the marginal over the input RandomVariable.
        
        This is a Factor whose scope is just [rvar]. Only well-defined
        when the model is calibrated.
        """
        clusters = self.clusters_with(rvar)
        if not clusters:
            raise ValueError("Random variable not found among this model's clusters.")
        #can sort by length of scope or w/e if this needs to be optimized
        cluster = clusters[0]
        undesired = set(cluster.scope) - {rvar}
        return Factor.marginalize(cluster.emit(), undesired)
    
class BayesNet:
    #TODO: take advantage of Model as a base class
    def __init__(self, cpds, rvar_labels):
        self._cpds = cpds
        self._rvars = rvar_labels.keys()
        self._adjacencies = Adjacency(
                            {rvar_labels[cpd.child] : [rvar_labels[parent]
                            for parent in cpd.parents] for cpd in cpds})
        self._adjacencies.reverse()
        self._graph = DirectedAcyclicGraph(self._adjacencies)
        self._joint = None
        
    @property
    def graph(self):
        return self._graph
    @property
    def joint(self):
        #TODO: fix this -- as written, could lead to a post-evidence bug
        if self._joint is None:
            self._joint = Factor.joint(*self._cpds)
        return self._joint
    
class CliqueTreeModel(GraphicalModel):
    """
    A CliqueTreeModel is a GraphicalModel that has an undirected tree graph
    representing the relationship between Clique nodes.
    
    Responsible for running exact inference over the Clique beliefs.
    """
    @staticmethod
    def generate(cliques):
        clique_tree = Graph.undirected()
        for clique in cliques:
            clique_name = ''.join(str(node) for node in clique)
            for other_clique in cliques:
                pass
        return None
    
    def __init__(self, named_clusters, adjacencies):
        super().__init__()
        self.attach(named_clusters)
        self._graph = AdjacencyGraph(adjacencies)
        self._scheduler = NodeScheduler()
    
    @property
    def clusters(self):
        return self._registry.values()
    @property
    def graph(self):
        return self._graph
    @property
    def schedule(self):
        return self._scheduler.topo_up_down(self.graph)
    
    def clusters_with(self, rvar):
        return [cluster for cluster in self.clusters if rvar in cluster.scope]
    def bp(self):
        """
        Belief propagation.
        """
        for node in self.schedule:
            sender = self.registry[node]
            for neighbor in self.graph.neighbors(node):
                receiver = self.registry[neighbor]
                message = sender.emit(excluded = {neighbor})
                receiver.receive(node, message)
    def marginal(self, rvar):
        """
        Return the marginal over the input RandomVariable.
        
        This is a Factor whose scope is just [rvar]. Only well-defined
        when the model is calibrated.
        """
        clusters = self.clusters_with(rvar)
        if not clusters:
            raise ValueError("Random variable not found among this model's clusters.")
        #can sort by length of scope or w/e if this needs to be optimized
        cluster = clusters[0]
        undesired = set(cluster.scope) - {rvar}
        return Factor.marginalize(cluster.emit(), undesired)

if __name__ == '__main__':
    pass
