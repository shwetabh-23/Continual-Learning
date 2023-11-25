# Continual-Learning
## Abstract :
This project focuses on sequential Named Entity Recognition (NER) in three distinct datasets, G1, G2, and G3, with the tasks labeled T1, T2, and T3. Each task involves identifying specific entities with different labels, like 'Treatment' and 'Cancer.' The learning process begins with independent training for each task, ensuring the model becomes proficient at recognizing entities unique to the dataset.
The key objective of this assignment is continual learning though, where the same model sequentially adapts to each new task while retaining knowledge from previous tasks. The fixed constraint of 100 examples from previous tasks helps to prevent catastrophic forgetting. This approach emphasizes knowledge retention, forward transfer (learning new tasks), and backward transfer (enhancing performance on previous tasks).
The project underscores the potential of continual learning in NER, offering a structured approach to address various NER tasks without compromising past knowledge. It has implications for real-world NER applications and the broader field of natural language processing.   
## Introduction : 
Named Entity Recognition (NER) is a fundamental natural language processing (NLP) task aimed at identifying specific entities within text, such as names of people, organizations, locations, and more. The main objective of this project is to perform NER across three datasets—G1, G2, and G3—each representing a unique NER task. These tasks, designated as T1, T2, and T3, share the same set of entity labels, encompassing categories like 'Treatment,' 'Cancer,' 'Chronic Disease,' 'Allergy,' and 'Other.' 
Continual learning, a powerful paradigm in machine learning, is employed to train a single model successively on these distinct NER tasks while keeping only a fixed number of training examples from the previous tasks. This approach exhibits several essential properties:
1.	Knowledge Retention: The model should safeguard the knowledge acquired during earlier tasks to avoid catastrophic forgetting.
2.	Forward Transfer: As the model tackles new NER tasks, it must reuse past knowledge to facilitate learning.
3.	Backward Transfer: Surprisingly, the model should display improved performance on tasks from its history after mastering new ones.
4.	Fixed Model Capacity: The model's memory size remains constant, irrespective of the number of tasks and data stream length.
By adhering to these properties, we aim to demonstrate the efficacy of our continual learning approach in NER tasks and assess its performance by creating test sets for each task. Additionally, we will compare the results with a model trained on the combined dataset (G1+G2+G3) to gauge the benefits of our structured continual learning paradigm.
## Data :
Our dataset comprises three distinct sources: G1, G2, and G3. Each dataset defines a unique Named Entity Recognition (NER) task—T1, T2, and T3—while sharing a common set of entity labels, including 'Treatment,' 'Cancer,' 'Chronic Disease,' 'Allergy,' and 'Other.' These datasets exhibit minimal overlap, fostering the creation of diversified NER challenges. 
To maintain consistency, we partitioned each dataset into training and test sets, preserving 80% for training and 20% for evaluation in each task. Our goal is to harness these datasets to train a single model continually across tasks, ensuring knowledge retention, forward and backward transfer, and a fixed model capacity while upholding NER performance standards.   
## Model Architecture : 
Our chosen model architecture for Named Entity Recognition (NER) tasks follows a deep learning approach, primarily composed of embedding layers, bidirectional Long Short-Term Memory (LSTM) units, and dense layers.
1.	Embedding Layer: To convert words into numerical representations, we utilize an embedding layer. This layer maps each word in the dataset to a continuous vector space. The embeddings are trainable, ensuring the model adapts to the specifics of each task.
2.	Bidirectional LSTM: Stacked bidirectional LSTM layers are employed to capture contextual information from both left and right sequences. These layers are vital for understanding dependencies among words and recognizing named entities.
3.	Dense Layer: A final dense layer with softmax activation assigns labels to each word, transforming the problem into a multi-class classification task.
## Model Evaluation : 
This is the most important aspect of the project. This table below shows how the model performed during its successive tests on different datasets. 
	Performance on the test set of T1	Performance on the test set of T1 and T2	Performance on the test set of T1 and T2 and T3	Performance
on combined
G1+G2+G3
Treatment
F1	0.78	0.76	0.79	0.79
Chronic
Disease
F1	0.80	0.82	0.84	0.01
Cancer
F1	0.78	0.80	0.84	0.81
Allergy
F1	0.00	0.00	0.00	0.0
Other
F1	1.00	1.00	0.99	0.07
Weight
averaged
F1	0.621	0.612	0.636	0.631

These values are fairly good considering the models are trained on a CPU machine with only 10 epoch of training and moderate complexity. Had it been trained on a better architecture, we would have surely seen better results.
Due to very low number of entities with the category “Allergy”, none of the models are able to learn to predict this category successfully. This is the reason why we get 0 f1-scores for this category and a significantly better score for other categories.

## Discussion and Conclusion :
 In this project, we explored a novel approach to Named Entity Recognition (NER) using Continual Learning. Our objective was to train a model for NER on three distinct datasets (T1, T2, and T3), each with its unique set of labels and entities. Additionally, we investigated the model's performance when trained using a traditional approach on a combined dataset (G1+G2+G3). The primary focus was on evaluating the properties of Continual Learning, such as knowledge retention, forward and backward transfer, and maintaining a fixed model capacity.

## Advantages of Continual Learning:
One of the key findings of this study is that Continual Learning exhibits several advantages over the traditional approach. First and foremost, it significantly reduces the training time as the model is trained sequentially on each task, eliminating the need for retraining the entire dataset simultaneously. This results in a more efficient use of computational resources.
Furthermore, Continual Learning ensures a fixed model capacity regardless of the number of tasks and data stream length. This property is vital in practical applications, as it allows the model to be deployed in memory-constrained environments.

## Performance and Future Directions:
We observed that the Continual Learning approach maintains knowledge retention, forward transfer, and even shows some backward transfer, which demonstrates its potential in real-world scenarios. However, there is room for improvement. We recommend exploring advanced model architectures like Attention Mechanisms and Transformers to further enhance NER performance.
In conclusion, our study highlights the potential benefits of Continual Learning in NER tasks, reducing training time and maintaining a fixed model capacity. It paves the way for more research into advanced model architectures and optimization techniques to achieve state-of-the-art performance in NER applications. Continual Learning proves to be a promising approach in addressing the evolving nature of NER tasks in the field of Natural Language Processing.
