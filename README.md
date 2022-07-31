# KI-CD

A knowledge infused Context driven (KI-CD) Disease Diagnosis VA


The repository contains the experimental setup, code and data for the paper A knowledge infused context driven dialogue agent for disease diagnosis using hierarchical reinforcement learning, Link : https://www.sciencedirect.com/science/article/pii/S0950705122000971 

The pre-print of the paper is available at 

A. main file  : KI-CD/src/dialogue_system/run/run.py

For DQN based dialogue agents :

	dqn_type = DQN

For DDQN based dialogue agents :

	dqn_type = DoubleDQN
  
For DDQN based dialogue agents :

	dqn_type = DuelingDQN


B.Other different varients of KI-CD


	1. KI-CD_HDC
	2. KI-CD_PCM
	3. KI-CD_MDD (KI-CD with the MDD dataset)
  

C.For Testing

	run_for_test.py

Please train before testing, there is not the saved model weight (higher, lower policies networks and disease classifier). 

## If you use the code, we appreciate it if you cite our paper
~~~~
@article{tiwari2022knowledge,
  title={A knowledge infused context driven dialogue agent for disease diagnosis using hierarchical reinforcement learning},
  author={Tiwari, Abhisek and Saha, Sriparna and Bhattacharyya, Pushpak},
  journal={Knowledge-Based Systems},
  volume={242},
  pages={108292},
  year={2022},
  publisher={Elsevier}
}
