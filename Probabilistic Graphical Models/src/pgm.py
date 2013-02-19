'''
Created on Feb 9, 2013

@author: Justin
'''

import scipy

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
    def register(self, observer):
        self._observers.add(observer)
    def unregister(self, observer):
        self._observers.discard(observer)
    def observe(self, *args, **kwargs):
        for observer in self._observers:
            observer._notify(*args, **kwargs)
        
    
class Observer:
    """
    Used as a base class for the Observer of the Observer Pattern.
    """
    def __init__(self, *args, **kwargs):
        pass
    def _notify(self, *args, **kwargs):
        raise NotImplementedError
    def register(self, subject):
        subject.register(self)
    def unregister(self, subject):
        subject.unregister(self)
        
    
class RandomVariable(Subject):
    """
    Each RandomVariable has a collection of distinct values
    representing the allowed instantiations of that random variable,
    commonly called the sample space or state space.
    
    A RandomVariable can be "observed" (in the scientific sense)
    in one of its states. If so, then it will notify all observers that
    its state has been set, but they must query to know what state that is.
    
    The RandomVariable is also responsible for knowing the size of its
    state space, i.e. its cardinality.
    
    Examples:
    (1) Binary random variable has state space (0, 1),
        a.k.a. coin-flip ('H', 'T')
    (2) The "classic" FICO score has a state space (300, 301, ..., 850)
        http://en.wikipedia.org/wiki/Credit_score_in_the_United_States#FICO_score_range
    
    In Physics, a RandomVariable is typically called a Local Field (Variable).
    
    ---FOR NOW: ONLY IMPLEMENT DISCRETE RANDOM VARIABLES---
    TODO: Implement continuous random variables
    """
    
    def __init__(self, states):
        """
        Construct a RandomVariable with the chosen state space,
        representing the possible realizations of some random variable.
        
        states -- the allowed instantiations of this random variable
        """
        super().__init__()
        if not isinstance(states, scipy.ndarray):
            raise NotImplementedError
        if states.ndim is not 1:
            raise NotImplementedError
        self._states = states
        self._cardinality = len(self._states)
        self._state = None
        self._index_of_state = None
    
#    def __repr__(self):
#        if self._state is None:
#            return 'RandomVariable({!r})'.format(self._states)
#        else:
#            return '{!r}'.format(self._state)
    def _index_of(self, s):
        try:
            return next(scipy.argwhere(self._states == s).flat)
        except StopIteration:
            return None
        
    @property
    def states(self):
        return self._states
    @property
    def cardinality(self):
        return self._cardinality
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, s):
        self._state = s
        self._index_of_state = self._index_of(s)
        self.observe(self)
    @property
    def index_of_state(self):
        return self._index_of_state
    

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
    
    ---FOR NOW: ONLY IMPLEMENT (1)---
    TODO: Implement (2) through (5)
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
        "TODO: Think about this design choice -- is this really what I want?"
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
        scope -- a collection of distinct random variables that
                    this factor has beliefs about
        beliefs  -- a scipy/numpy array (possibly 0-rank) of real numbers with
                    (1) beliefs.ndim == len(scope)
                    (2) beliefs.shape[k] == scope[k].cardinality for all k
                    (3) Order of axes should reflect the order of random variables
                        in the scope!
                        
        N.B. If 'scope' is empty and 'beliefs' is scipy.array(0) or scipy.array(1), these
        represent the 'identity factors' for addition and multiplication respectively.
        """
        super().__init__()
        if not isinstance(beliefs, scipy.ndarray):
            raise NotImplementedError("'beliefs' must be a scipy/numpy array.")
        assert all(isinstance(belief, (scipy.integer, scipy.floating))
                   for belief in beliefs.flat), \
                "The beliefs must all be real numbers."
        assert beliefs.ndim == len(scope), \
                "The beliefs do not match the scope."
        assert all(isinstance(rvar, RandomVariable) for rvar in scope), \
                "The scope must be a collection of random variables."
        assert all(num_states == rvar.cardinality
                   for (num_states, rvar) in zip(beliefs.shape, scope)), \
                "The beliefs do not match the cardinalities of the random variables."
        assert hasattr(scope, 'index'), \
                "The scope must be an indexed collection."
        self._scope = scope
        self._beliefs = beliefs
        for rvar in self._scope:
            self.register(rvar)
        
#    def __mul__(self, other):
#        return Factor.product(self, other)
#    def _axis_of(self, rvar):
#        return self._scope.index(rvar)
    def _proxify(self):
        self._proxy_scope = [rvar for rvar in self._scope if rvar.state is None]
        subslice = tuple(slice(None) if rvar.state is None else rvar.index_of_state
                    for rvar in self._scope)
        #makes subtle use of [()] indexing in case belief is 0-rank
        #self._beliefs is -always- an ndarray, whereas
        #self._proxy_beliefs is often an ndarray, but may be a scalar,
        #typically when self._proxy_scope is empty
        self._proxy_beliefs = self._beliefs[subslice]
        
    @_mutates_proxies()
    def _notify(self, rvar):
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


class Model:
    """
    A Model is a collection of RandomVariables together with Factors that encode
    beliefs about (subsets of) those variables and their interdependencies.
    
    Important examples include:
    (1) Bayesian Networks
    (2) Markov Random Fields (including Ising Models, Restricted Boltzmann Machines)
    (3) Non-graphical Models (?)
    
    ---FOR NOW: ONLY IMPLEMENT (1) and (2) ---
    """
    
    def __init__(self, params):
        """
        Constructor
        """