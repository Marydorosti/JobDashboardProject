# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 08:47:02 2022

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:30:57 2021

@author: m.dorosti
"""

from deap import base
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import elitism
#import nurses
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import traceback
import sys
from flask import Flask, request, render_template, session, redirect
import statistics
import json
from deap import algorithms

app = Flask(__name__,template_folder='Template')

class Knapsack01Problem:
    """This class encapsulates the Knapsack 0-1 Problem from RosettaCode.org
    """

    def __init__(self,items,maxCapacity,hardConstraintPenalty=60):

        # initialize instance variables:
        #self.items = []
        #self.maxCapacity = 0
        self.items=items
        self.maxCapacity=maxCapacity
        self.hardConstraintPenalty=60

        # initialize the data:
        #self.__initData()

    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return len(self.items)

    

    def getValue(self, attr_fltList):
        """
        Calculates the value of the selected items in the list, while ignoring items that will cause the accumulating weight to exceed the maximum weight
        :param zeroOneList: a list of 0/1 values corresponding to the list of the problem's items. '1' means that item was selected.
        :return: the calculated value
        zeroOneList[i]
        """

        totalWeight = totalValue = 0

        for i in range(len(attr_fltList)):
            violation=0
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                totalWeight +=   attr_fltList[i] * weight
                totalValue +=   attr_fltList[i] * value
            #if totalWeight >= self.maxCapacity:
               # violation=violation+1
        return totalValue,violation
    
    def evaluate(self,attr_fltList):
        
        
        violation=0
        totalWeight = totalValue = 0
        if  statistics.stdev(attr_fltList)<=20:
                #violation=violation+10000
        
        
        

          for i in range(len(attr_fltList)):
            violation=0
            item, weight, value = self.items[i]
            if totalWeight +attr_fltList[i]*weight <= self.maxCapacity:
                totalWeight +=   attr_fltList[i] * weight
                totalValue +=   attr_fltList[i] * value
            else:
                totalValue=0
            
        #if totalWeight >= self.maxCapacity:
               # violation=violation+1   
        else:
            totalValue=0
    
        return totalValue       
            
        #return (1+(0.5*totalValue)-(1+(0.5*violation)))
        
        
        
            
             
    def getCost(self, attr_fltList):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """

       
        hardContstraintViolations =  self.Violation2(attr_fltList)
    
        
        return  hardContstraintViolations
        
        
        
        
        
        

    def printItems(self, attr_fltList):
        """
        Prints the selected items in the list, while ignoring items that will cause the accumulating weight to exceed the maximum weight
        :param zeroOneList: a list of 0/1 values corresponding to the list of the problem's items. '1' means that item was selected.
        """
        totalWeight = totalValue = 0
        j=[]

        for i in range(len(attr_fltList)):
            item, weight, value = self.items[i]
           # if attr_fltList <= self.maxCapacity:
           # if attr_fltList[i]<= self.maxCapasityOfeachItem[i] :   
            if totalWeight + attr_fltList[i] *weight <= self.maxCapacity:
                
                if attr_fltList[i] > 0:
                    totalWeight += attr_fltList[i] *weight
                    totalValue += attr_fltList[i] *value
                    c="- Adding {}: weight = {}, value = {}, accumulated weight = {}, accumulated value = {}".format(item, weight, value, totalWeight, totalValue)
                    j.append(c)
        d="- Total weight = {}, Total value = {}".format(totalWeight, totalValue)
        #j=[] 
        #j.append(c) 
        j.append(d)
        return(j)                                               

@app.route('/main', methods=['POST','GET']) 

def main():
    json_ = request.json 
    maxCapacity=json_["maxCapacity"]
    items=json_["items"]
    #hardConstraintPenalty=json_["hardConstraintPenalty"]
    
    knapsack = Knapsack01Problem(items=items,maxCapacity=maxCapacity)

# Genetic Algorithm constants:
    POPULATION_SIZE = 50
    P_CROSSOVER = 0.9  # probability for crossover
    P_MUTATION = 0.1   # probability for mutating an individual
    MAX_GENERATIONS =1000
    HALL_OF_FAME_SIZE = 1
    


# set the random seed:
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    toolbox = base.Toolbox()
    import numpy as np

    toolbox.register("attr_flt", random.randint,5,40)

# define a single objective, maximizing fitness strategy:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))


    creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.attr_flt, len(knapsack))

# create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation
    def knapsackValue(individual):
       return knapsack.getValue(individual);# return a tuple
    def knapsackCost(individual):
       return knapsack.getCost(individual);# return a tuple

    def evaluate(individual):
       return  knapsack.evaluate(individual),

    toolbox.register("evaluate",evaluate)

# genetic operators:mutFlipBit

# Tournament selection with tournament size of 3:
    toolbox.register("select", tools.selTournament, tournsize=3)

# Single-point crossover:
    toolbox.register("mate", tools.cxTwoPoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(knapsack))


# Genetic Algorithm flow:


    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    
            
        
        

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
   
    # print best solution found:
    best = hof.items[0]
    d="-- Best Ever Individual = ", best
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    print("-- Knapsack Items = ")
    j=knapsack.printItems(best)
     #q=nsp.printItems(best)
    date=datetime.now()
    EndDate=str(EndDate[0])+"-"+str(EndDate[1])+"-"+str(EndDate[2])

    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    return  '{} {}'.format(d,j)
    #return   str(j),str(d)


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 13845 # If you don't provide any port the port will be set to 12345

    

    

    app.run(port=port, debug=True)























