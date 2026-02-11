from base.agent import QRNAgent
import numpy
numpy.set_printoptions(legacy='1.25')


if __name__ == '__main__':

	agent = QRNAgent(buffer_size=10_000)

	agent.train(episodes=4000, 
			 	max_steps=80, 
			 	savemodel=True,
			 	plot=True, 
			 	savefig=True,
			 	jitter=None, 
			 	n_range=[4, 4], 
			 	p_e=0.50, 
			 	p_s=0.99,
			 	tau=50,
			 	cutoff=15)
	