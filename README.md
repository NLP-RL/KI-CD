# KI-CD

A knowledge infused Context driven (KI-CD) Disease Diagnosis VA



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

Please train before testing, there is not the saved model weight (higher, lower policies networks and disease classifier) as its higher size did not allow it to be uploaded.
