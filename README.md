# Subphenotyping-for-PASC
This demo code is for our manuscript "Machine Learning for Identifying Data-Driven Subphenotypes of Incident Post-Acute SARS-CoV-2 Infection Conditions with Large Scale Electronic Health Records: Findings from the RECOVER"

If there is any questions, please contact: zhanghao.duke@gmail.com

Step 1 Data preparation

Since the privacy of dataset we used in the study, in this demo code, we just provide simulated dataset. 
For the real dataset, The INSIGHT data can be requested through https://insightcrn.org/. The OneFlorida+ data can be requested through https://onefloridaconsortium.org. Both the INSIGHT and the OneFlorida+ data are HIPAA-limited. Therefore, data use agreements must be established with the INSIGHT and OneFlorida+ networks. 

After data preparation from the raw EHR tabel, we can obtain a data matrix with size N*137. N is the number of patients in the corhot, and 137 denotes 137 PASC. This is a binary matrix, where the element in i-th raw, j-th column denotes whether i-th patient has j-th pasc in the post-acute SARS-CoV-2 infection period of COVID-19
We put the simulated dataset in "https://drive.google.com/file/d/1ZN_hIiDfazCHOGl1GHQMNBXEkwk5a_TH/view?usp=sharing". After downloading, please put it in the the folder: "./dataset/"

Step 2 Train the Topic model

We prepared both Python and Matlab code for training the topic model

For Python,
please run "./Python code for training topic modeling/Main_train_topic_model.py"

For Matlab,
please run "./Matlab code for training topic modeling/main_PFA.m"

The parameter K in both codes are the nubmer of topics. In our study, we set it as 10.

After training, the well-trained model (Topics and topic proportions) are saved in "./trained_topic_model/".


Step 3 Visualize the topic

To reproduce the figure 2 in our manuscipt, please refer the code 
"./Python code for analysis/visualize_topic.py"

Step 4 Perform hierarhical clustering

To identify the subphenotypes based on topic proportions, please run the code "./Python code for analysis/perform_clustering.py". It will also reproduce the Supplemental figure 5 (UMAP and dendrogram).

To reproduce the Figure 3, please run "./Python code for analysis/show_circle_plot_PASC.py"

To reproduce the Figure  4, please run ""./Python code for analysis/show_circle_plot_med.py"

