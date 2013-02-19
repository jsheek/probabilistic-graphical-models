'''
Created on Feb 12, 2013

@author: Justin
'''

import unittest
import pgm
import scipy

class Test(unittest.TestCase):
    
    def setUp(self):
        self.states_list = []
        self.raw_beliefs = []
        self.shapes = []
        self.factors = []
        self.states_list.append(scipy.array([0, 1]))
        self.states_list.append(scipy.array([0, 1]))
        self.states_list.append(scipy.array([0, 1]))
        self.raw_beliefs.append([0.11, 0.89])
        self.raw_beliefs.append([0.59, 0.41, 0.22, 0.78])
        self.raw_beliefs.append([0.39, 0.61, 0.06, 0.94])
        self.shapes.append((2))
        self.shapes.append((2, 2))
        self.shapes.append((2, 2))
        
        self.beliefs = [scipy.reshape(belief, shape, order = 'F') for \
                        (belief, shape) in zip(self.raw_beliefs, self.shapes)]
        self.rvars = [pgm.RandomVariable(states) for states in self.states_list]
        
        self.factors.append(pgm.Factor([self.rvars[0]], self.beliefs[0]))
        self.factors.append(pgm.Factor([self.rvars[1], self.rvars[0]], self.beliefs[1]))
        self.factors.append(pgm.Factor([self.rvars[2], self.rvars[1]], self.beliefs[2]))
        
    def tearDown(self):
        pass
        
    def testOrdering(self):
        for (factor, belief) in zip(self.factors, self.raw_beliefs):
            assert scipy.allclose(factor.beliefs.ravel(order = 'F'), belief)
        
    def testReordering(self):
        pass
    
    def testObservation(self):
        for (k, state) in enumerate(self.states_list[0]):
            self.rvars[0].state = state
            assert (self.factors[0].beliefs == self.beliefs[0][k]).all()
            assert (self.factors[1].beliefs == self.beliefs[1][:, k]).all()
            assert (self.factors[2].beliefs == self.beliefs[2]).all()
        self.rvars[0].state = None
        
        for (k, state) in enumerate(self.states_list[1]):
            self.rvars[1].state = state
            assert (self.factors[0].beliefs == self.beliefs[0]).all()
            assert (self.factors[1].beliefs == self.beliefs[1][k, :]).all()
            assert (self.factors[2].beliefs == self.beliefs[2][:, k]).all()
        self.rvars[1].state = None
        
        for (k, state) in enumerate(self.states_list[2]):
            self.rvars[2].state = state
            assert (self.factors[0].beliefs == self.beliefs[0]).all()
            assert (self.factors[1].beliefs == self.beliefs[1]).all()
            assert (self.factors[2].beliefs == self.beliefs[2][k, :]).all()
        self.rvars[2].state = None
        
    def testMarginalization(self):
        self.marginals = []
        self.marginals.append([1])
        self.marginals.append([1, 1])
        self.marginals.append([1, 1])
        for (factor, rvar, marginal) in zip(self.factors, self.rvars, self.marginals):
            actual_marginal = pgm.Factor.marginalize(factor, [rvar]).beliefs
            assert scipy.allclose(actual_marginal, marginal)
    
    def testMultiplication(self):
        self.raw_derived_beliefs = []
        self.derived_shapes = []
        self.derived_factors = []
        self.raw_derived_beliefs.append([0.0649, 0.1958, 0.0451, 0.6942])
        self.raw_derived_beliefs.append([0.025311, 0.076362, 0.002706, 0.041652, \
                                  0.039589, 0.119438, 0.042394, 0.652548])
#        self.raw_derived_beliefs.append([0.025311, 0.076362, 0.002706, 0.041652, \
#                                  0.039589, 0.119438, 0.042394, 0.652548])
        self.derived_shapes.append((2, 2))
        self.derived_shapes.append((2, 2, 2))
        self.derived_shapes.append((2, 2, 2))
        self.derived_beliefs = [scipy.reshape(belief, shape, order = 'F') for \
                                (belief, shape) in zip(self.raw_derived_beliefs, self.derived_shapes)]
        self.derived_factors.append(pgm.Factor.product(self.factors[0], self.factors[1]))
        self.derived_factors.append(pgm.Factor.product(self.derived_factors[0], self.factors[2]))
#        self.derived_factors.append(pgm.Factor.joint(*self.factors))
        for (factor, belief) in zip(self.derived_factors, self.derived_beliefs):
            assert scipy.allclose(factor.beliefs, belief)
    
    def testEvidenceJointMarginalWorkflow(self):
        self.raw_probabilities = [0.0858, 0.0468, 0.1342, 0.7332]
        self.probabilities = scipy.reshape(self.raw_probabilities, (2, 2), order = 'F')
        self.reduced_probabilities = scipy.sum(self.probabilities, axis = 0)
        #Evidence
        self.rvars[0].state = 1
        #Joint
        self.joint_factor = pgm.Factor.joint(*self.factors)
        #Marginal
        self.reduced_factor = pgm.Factor.marginalize(self.joint_factor, [self.rvars[1]])
        assert scipy.allclose(self.reduced_factor.probabilities, self.reduced_probabilities)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()