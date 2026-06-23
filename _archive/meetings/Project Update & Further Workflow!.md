**Project Update & Further Workflow\!**

**Date:** March 13, 2026 7:30 PM IST  
**Meeting Recording:** [See full transcript](https://app.fireflies.ai/view/01KKK0KV4N7E4DXJ85SK5F260K)  
**Participants:** pramod.kachare@rait.ac.in, maheshkaachyut@gmail.com, ruchitdas36@gmail.com, 048niharika@gmail.com, sandeep.sangle@rait.ac.in, mruduljadhav18@gmail.com  
**Duration:** 81 mins

**Keywords**

Parkinson’s disease, dataset preprocessing, RMS normalization, feature extraction, machine learning evaluation, speech analysis

**Overview**

**•  Data Set Review:** Consolidated six Parkinson’s speech data sets, totaling 7,632 files, including balanced samples of healthy and Parkinson’s speakers.    
**• Preprocessing Strategy:** All audio files to be resampled to 8,000 Hz to ensure uniformity and avoid clipping.    
**• Model Evaluation Plan:** Established a seven-by-seven confusion matrix to assess model performance across individual and combined data sets.    
**• Modular Coding Design:** Proposed a modular approach for processing and training, enabling flexible data combinations and automated runs.    
**• Foundation Model Development:** Focus on combining diverse data sets, starting with sustained vowels for robust modeling across populations.

**Notes**

**📊 Literature Survey and Feature Analysis**  
• Presented by Achyut Maheshka  
• Included data sources, audio file types, languages, and classifier accuracies

**📂 Dataset Discussion**  
• Discussion on restricted and open datasets  
• Confirmation of downloaded SAGA 2019 audio files  
• Need for access verification for restricted datasets

**🔍 DOI Mismatches**  
• Identification of mismatches in dataset DOIs  
• Some DOIs correspond to unrelated research  
• Corrections needed for accuracy

**✅ Completed Datasets**  
• Confirmation of completed datasets: PC Gita, MDVR KCL  
• Some datasets pending full analysis or download

**🗣️ Speech Types Clarification**  
• Clarification on DDK task for motor speech assessment  
• Importance for Parkinson’s evaluation

**📈 Feature Review**  
• Review of extracted features  
• Categorized into articulation, formants, temporal and spectral features

**🔄 Sampling Frequency Unification**  
• Decision to unify sampling frequencies  
• Downsampling to \~8 kHz  
• RMS normalization to fix amplitude discrepancies

**⚙️ RMS Energy Calculation**  
• Explanation on calculating RMS energy  
• Normalization approach and target values

**🔗 Dataset Combination Plan**  
• Plan for combining datasets post-preprocessing  
• Modularized feature extraction from grouped data

**📊 Evaluation Framework Development**  
• Comprehensive evaluation framework with confusion matrices  
• 7x7 matrix setup for model performance assessment

**🤖 Machine Learning Models Proposal**  
• Proposal to test Random Forest and Gradient Boosting  
• Focus on modular code development

**🛠️ ML Tools Suggestion**  
• Utilize Orange for fast ML model evaluation  
• Emphasize modularized coding for scalability

**📅 Communication Protocol**  
• Queries via WhatsApp group  
• Schedule follow-up meetings after initial tasks

**Action Items**

**Achyut Maheshka:**  
• Verify and correct dataset DOIs, especially those mismatched for Parkinson’s data.  
• Prepare and share updated CSV or list highlighting datasets pending download or verification.  
• Provide detailed dataset content summary (number of speakers, speech types, audio file counts) for PC Gita and others using pivot tables or summaries.  
• Extract features for all available datasets modularly and prepare the processed dataset for training/testing.  
• Update project lead after completing the first dataset processing phase.

**Niharika Mishra:**  
• Assist in data exploration, pivot table generation, and feature summary visualization.  
• Clarify DDK speech type and its relevance for Parkinson’s assessment.  
• Forward corrected and updated dataset details and features to the lead and team.

**Pramod Kachare:**  
• Obtain restricted access to datasets on Zenodo and complete data usage agreements.  
• Provide modularized code or computational resources (machines) for running feature extraction and classification experiments.  
• Guide the team on preprocessing (downsampling, RMS normalization), modular coding structure, feature extraction, and experimental design including confusion matrix evaluation strategy.  
• Review early results and plan follow-up meetings based on day-one completion progress.

**Sandeep Sangle:**  
• Coordinate task division focusing on initial feature extraction from datasets before classification.  
• Recommend use of Orange tool for efficient ML experimentation post feature extraction.  
• Facilitate quick testing cycles for classification models after modularization.

