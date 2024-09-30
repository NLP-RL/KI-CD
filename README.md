### A knowledge infused context driven dialogue agent for disease diagnosis using hierarchical reinforcement learning

The repository contains the code for our research work titled 'A knowledge infused context driven dialogue agent for disease diagnosis using hierarchical reinforcement learning'. 

### Abstract
Disease diagnosis is an essential and critical step in any disease treatment process. Automatic diagnostic testing has gained popularity in recent years due to its scalability, rationality, and efficacy. The major challenges for the diagnosis agent are inevitably large action space (symptoms) and varieties of diseases, which demand either rich domain knowledge or an intelligent learning framework. We propose a novel knowledge-infused context-driven (KI-CD) hierarchical reinforcement learning (HRL) based diagnosis dialogue system, which leverages a bayesian learning-inspired symptom investigation module called potential candidate module (PCM) for aiding context-aware, knowledge grounded symptom investigation. The PCM module serves as a context and knowledge guiding companion for lower-level policies, leveraging current context and disease-symptom knowledge to identify candidate diseases and potential symptoms, and reinforcing the agent for conducting an intelligent and context guided symptom investigation with the information enriched state and an additional critic known as learner critic. The knowledge-guided symptom investigation extracts an adequate set of symptoms for disease identification, whereas the context-aware symptom investigation aspect substantially improves topic (symptom) transition and enhances user experiences. Furthermore, we also propose and incorporate a hierarchical disease classifier (HDC) with the model for alleviating symptom state sparsity issues, which has led to a significant improvement in disease classification accuracy. Experimental results (both quantitatively and qualitatively) on the benchmarked dataset establish the need and efficacy of the proposed framework. The proposed framework outperforms the current state-of-the-art method in all evaluation metrics other than dialogue length (diagnosis success rate, average match rate, symptom identification rate, and disease classification accuracy by 7.1 %, 0.23 %, 19.67 % and 8.04 %, respectively).  

![Working](https://github.com/NLP-RL/KI-CD/blob/main/KICD.png)

#### Full Paper: https://www.sciencedirect.com/science/article/pii/S0950705122000971 

### A. KI-CD  : KI-CD/src/dialogue_system/run/run.py

For DQN based dialogue agents :

	dqn_type = DQN

For DDQN based dialogue agents :

	dqn_type = DoubleDQN
  
For DDQN based dialogue agents :

	dqn_type = DuelingDQN

### B.Other varients of KI-CD
	1. KI-CD_HDC
	2. KI-CD_PCM
	3. KI-CD_MDD (KI-CD with the MDD dataset)
  
### C.For Testing
	run_for_test.py


## Citation Information

If you find this code useful in your research, please consider citing:
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

Please contact us @ abhisektiwari2014@gmail.com for any questions, suggestions, or remarks. 
