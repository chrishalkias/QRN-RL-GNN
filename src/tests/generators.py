import random

def generateRandom_N():
    return random.randint(3,30)

def generateRandom_pe():
    return random.uniform(0.001, 0.99)

def generateRandom_ps():
    return random.uniform(0.1, 0.99)

def generateRandom_tau():
    return random.randint(100, 1000)

def generateRandom_cutoff():
    return random.randint(100, 1000)