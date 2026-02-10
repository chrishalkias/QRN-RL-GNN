from base.agent import QRNAgent
import numpy
numpy.set_printoptions(legacy='1.25')


if __name__ == '__main__':

	agent = QRNAgent(buffer_size=10_000)

	agent.train(episodes=500, 
			 	max_steps=100, 
			 	savemodel=True,
			 	plot=True, 
			 	savefig=True,
			 	jitter=500, 
			 	n_range=[4, 6], 
			 	p_e=0.99, 
			 	p_s=0.95,
			 	tau=1000,
			 	cutoff=100)
	