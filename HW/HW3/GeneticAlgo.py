import numpy as np

def recipe_success(state):
	fitness = 0
	
	# problem specific
	target = [36, 36, 48, 1, 1, 108, 1, 0.5, 96]
	
	i = 0
	for gs, gt in zip(state, target): 
		if gs != gt: 
### 1			
#			fitness+=1
### 2			
#			x = np.abs(gs - gt)
#			if x < 10: fitness += 1
#			elif x < 20: fitness += 2
#			elif x < 30: fitness += 3
#			else: fitness += 5
### 3			
#			fitness += 100 * np.abs((gs/108) - (gt/108))
### 4			
#			fitness += 100 * np.abs((gs/target[i]) - (gt/target[i]))
#			i+=1
### 5
#			x = np.abs((gs/target[i]) - (gt/target[i]))
#			if x < 0.1: fitness += 1
#			elif x < 0.2: fitness += 2
#			elif x < 0.3: fitness += 3
#			elif x < 0.4: fitness += 4
#			elif x < 0.5: fitness += 5
#			elif x < 0.6: fitness += 6
#			elif x < 0.7: fitness += 7
#			elif x < 0.8: fitness += 8
#			elif x < 0.9: fitness += 9
#			else: fitness += 10
#			i+=1
### 6: MY BEST ONE
			x = np.abs((gs/target[i]) - (gt/target[i]))
			if x < 0.05: fitness += 1
			elif x < 0.1: fitness += 2
			elif x < 0.15: fitness += 3
			elif x < 0.2: fitness += 4
			elif x < 0.25: fitness += 5
			elif x < 0.3: fitness += 6
			elif x < 0.35: fitness += 7
			elif x < 0.4: fitness += 8
			elif x < 0.45: fitness += 9
			elif x < 0.5: fitness += 10
			elif x < 0.55: fitness += 11
			elif x < 0.6: fitness += 12
			elif x < 0.65: fitness += 13
			elif x < 0.7: fitness += 14
			elif x < 0.75: fitness += 15
			elif x < 0.8: fitness += 16
			elif x < 0.85: fitness += 17
			elif x < 0.9: fitness += 18
			elif x < 0.95: fitness += 19
			else: fitness += 20
			i+=1
	return fitness
	
# Your code here.
class problem:
	
	def __init__(self, dna_length, population_length, objective_function, mutation_probability, fitness_goal):
		'''
		initial_population = list of lists; each sub-list is a dna string for a population member
		objective_function = objective function to maximize
		mutation_probability = probability that any given child has a mutation
		fitness_goal = fitness goal to achieve (stopping criterion, once member reaches this)
		
		In our case, I've chosen to minimize this fitness_goal at 0
		This felt natural for numerous reasons and I cannot necessarily think of an example when this would be helpful to maximize
		'''
		self.n_pop = population_length
		self.n_dna = dna_length
		self.initial_population = self.create_gnome()
		self.population = self.initial_population
		self.objective_function = objective_function
		self.p_mutate = mutation_probability
		self.fitness_goal = fitness_goal

	def mutate_gene(self, gene=None): 
		''' 
		create random genes for mutation 
		
		problem specific: 
		(really only works on this problem with lists of ints)
		lots of trial and tribulation here...
		''' 
		
		# initialization: gene=None 
		# return a gene in the valid range for the problem
		if gene == None:
			return np.random.choice(np.arange(0, 108.5, 0.5))
			
			
		# else 
		
		# choose if the gene will be randomly mutated (sorta aids like a random restart in hillclimbing)
		random_mutate = np.random.choice([True, False], p=[self.p_mutate, 1-self.p_mutate])
		
		# randomly mutate if chosen
		if random_mutate:
			return np.random.choice(np.arange(0, 108.5, 0.5))
		
		# else
		
		# choose if the plus minus will be high or low	
		high_pm = np.random.choice([True, False], p=[self.p_mutate, 1-self.p_mutate])
		
		# we choose our plus minus 	
		if high_pm:
			plus_minus = 2.5
		else:
			plus_minus = 1
		
		# get random value x in range [-plus_minus, plus_minus] such that adding the value x to gene doesn't invalidate it
		x = 0
		while x == 0:
			x = np.random.choice(np.arange(-plus_minus, plus_minus+0.5, 0.5))
			if (gene + x < 0) or (gene + x > 108.5):
				x = 0
		# finally return said value x with the gene
		return gene + x
		

	def create_gnome(self): 
		''' 
		create chromosome or string of genes 
		'''
		# very straightforward, creating the gnome problem specific
		population = []
		for _ in range(self.n_pop):
			population.append([self.mutate_gene() for _ in range(self.n_dna)])
		return population 

	def mate(self, par1, par2): 
		''' 
		Perform mating and produce new offspring 
		'''
		# chromosome for offspring 
		child_chromosome = [] 
		
		# computing probabilties for each parents gene being chosen given the mutation probability
		p2_prob = 1 - self.p_mutate
		p1_prob = p2_prob / 2
		for gp1, gp2 in zip(par1, par2):     

				# random probability   
				prob = np.random.random() 
				p2_prob = 1 - self.p_mutate
				p1_prob = p2_prob / 2
				
				# if prob is less than 0.45, insert gene 
				# from parent 1  
				if prob < p1_prob: 
					child_chromosome.append(gp1) 

				# if prob is between 0.45 and 0.90, insert 
				# gene from parent 2 
				elif prob < p2_prob: 
					child_chromosome.append(gp2) 

				# otherwise insert random gene(mutate),  
				# for maintaining diversity 
				else: 
					child_chromosome.append(self.mutate_gene(int((gp1 + gp2) / 2))) 
#		print(child_chromosome)
		# generated chromosome for offspring 
		return child_chromosome

def genetic_algorithm(problem, n_iter):

	generation = 1
	pop = problem.population

	for _ in range(n_iter-1):

		# sort the population in increasing order of fitness score 
		pop = sorted(pop, key = lambda x: problem.objective_function(x)) 
		
		# if the individual having lowest fitness score ie.  
		# 0 then we know that we have reached to the target 
		# and break the loop 
		if problem.objective_function(pop[0]) <= 0: 
			print("\nMATCH FOUND\n!!!Generation: {}\tString: {}\tFitness: {}".format(generation, pop[0], problem.objective_function(pop[0])))
			return

		# Otherwise generate new offsprings for new generation 
		new_generation = [] 

		for _ in range(problem.n_pop):
			h = int(len(pop) / 2) 
			indices = np.arange(h)
			rnd_indices = np.random.choice(h, size=2)
			child = problem.mate(pop[-rnd_indices[0]], pop[-rnd_indices[1]]) 
			new_generation.append(child) 
			
		
		new_generation = sorted(new_generation, key = lambda x: problem.objective_function(x))
		
		s = int((10*problem.n_pop)/100) 
		new_generation = new_generation[:s]		
		
		# Perform Elitism, that mean 10% of fittest population 
		# goes to the next generation 
		new_generation += pop[:s]

		# From 50% of fittest population, Individuals  
		# will mate to produce offspring 
		s = int((90*problem.n_pop)/100) 

		pop = new_generation 

		print("Generation: {}\tString: {}\tFitness: {}".format(generation, pop[0], problem.objective_function(pop[0]))) 

		generation += 1


	print("\nMATCH NOT FOUND...\nGeneration: {}\tString: {}\tFitness: {}".format(generation, pop[0], problem.objective_function(pop[0])))

genetic_problem = problem(dna_length=9,
						  population_length=10, 
						  fitness_goal=0, 
						  mutation_probability=0.1, 
						  objective_function=recipe_success)
out = genetic_algorithm(genetic_problem, 200**2)
print(out)