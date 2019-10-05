---
layout: post
title: Opinion Mining of Big data for Digital Governance
type: python
comments: true
categories: project
tags: [PYTHON]
img: social.jpg
---

## Table of Content
1. Introduction  
	1.1 Motivation & Scope    
	1.2 Basic Terminology  

2. Research Methodology   
	2.1 Systematic Literature Review  
	2.2 Research Gaps  

3. Research Objective  

4. Overview of Work   

5. Opinion Mining Model for Government Policy Evaluation  
	5.1 Three dimensions of Opinion Mining    
	5.2 Techniques Used In Proposed Model  
		5.2.1 Opinion Mining using Machine Learning  
	 	5.2.2 Opinion Mining using Hybrid Techniques  
	5.3 Tasks performed in Proposed model  
	5.4 Opinion Mining Model  

6. Application Area in Proposed Model     

---

<center><h1>1. INTRODUCTION</h1></center> 

As the web technologies have evolved, participation & communication have been identified as two pro-aspects that enable us to find out opinions within the vast pool of virtually connected people, who are neither personal acquaintances nor well known professional critics. More specifically the advent of real time sites has instigated the creation of an unparallel public collection of opinions about every object of interest. There have been several ongoing research projects that apply sentiment analysis to the variety of web based corpora in order to extract general public sentiment or opinion regarding political issues, product reviews, government policy making, market intelligence etc. Literature survey in the proposed research area identified it as a potential dynamic direction of research with promising applications and technologies.

## 1.1 Motivation and Scope

Opinion Mining or Sentiment analysis on big data has emerged as a notable direction of research with scientific trials and promising applications being explored substantially. It has turned out as an exciting new trend with a gamut of practical applications that range from Business Intelligence; Recommender Systems; Expert Finding; Politics; Advocacy Monitoring, amongst others.

The thrust area is to explore and understand the opinion mining framework, goals and its application on big data for the purpose of digital governance. The requirement is to go through a state-of-the-art literature survey and review the significant research on the subject of sentiment analysis, expounding its basic terminology and tasks. The idea is to envision “digital governance” by virtue of social web adoption, where government can take advantage of social platforms involving huge user participation.

The motivation is clearly based on Digital India Initiative, a national flagship programme proposed & projected by Government of India. The initiative is a big step to transform the country into a digitally empowered knowledge economy & includes projects that aim to ensure that government services are available to citizens electronically and people get benefit of the latest information and communication technology.

## 1.2 Basic Terminology

**_Sentiment Analysis_** - Sentiment analysis, as it sometimes called, is one of many areas of computational studies that deal with opinion oriented natural language processing. Such opinion oriented studies include among others, genre distinctions, emotion and mood recognition, relevance computations, perspectives in text, text source identification and opinion oriented summarization. Sentiment Analysis is a field of study that tends to use the natural language processing techniques to extract, capture or identify the viewpoint of a person with respect to a particular subject. It is the automated mining of attitudes, opinions, and emotions from text, speech, and database sources through NLP. The primary task is to opinionate, i.e. to sort and categorize one's perspective into positive, negative or neutral views. Once it is done, it can be further sub categorized into two parts, one of them focusing on the information that is factual, more likely to be an objective description of a unit, while the other emphasizes on sentiments, that are subjective in the expression of the feelings of the opinion holder. Both of them hold equal importance in deducing conclusions.

**Approaches of Sentiment Analysis**  
Analyzing the content of social media for opinion mining is a tedious task as it requires a thorough and extensive knowledge of the rules associated with the NLP. For e.g. syntactical and semantic, the explicit and implicit, regular and irregular language rules. Three main techniques used for opinion polarity classification are as follows:

![a](/assets/img/om_techniques.png){:class="img-responsive"}
<center><b>Figure 1. Approaches of Sentiment Analysis</b></center>

- **Machine Learning Approach** - The machine learning approaches can be grouped into: supervised and unsupervised learning methods. In supervised method, learning done from training data is applied to a new test data, whereas, in unsupervised method there is no prior learning (i.e. no training data), and the task is to find hidden structure in unlabeled data.

- **Lexicon Based Approach** - The lexicon-based approaches tend to be dependent on the sentiment vocabulary, that provides a collection of known and precompiled sentiment terms. It is also further divided into two categories, namely the dictionary-based approaches and the corpus based approaches which use statistical or semantic methods to find sentiment polarity and determine the emotional affinity of words, which is to learn their probabilistic affective scores from large corpora.

- **Hybrid Approach** - The hybrid approach is the combination of both the above mentioned methods and plays an important role in decision making as the techniques of both the approaches are collaborated for a better result.

**_Digital Governance_** - Governance is all about to conduct, direct, exercise power and implement the practices in order to ensure the aura of an organization/system is well-suited, vigorous and effective. It is a mechanism of formation of decisions and their implementation which constitutes usability, applicability and steering the resources of a country.

At the same time, good democratic governance is to establish, consensus amongst the stakeholders of a country with the goal to improve quality of life enjoyed by all citizens. In a resolution 2000/64 the commission of human rights (United Nations Human Rights) identified the key attributes of good governance:

- Transparency  
- Responsibility  
- Accountability  
- Participation  
- Responsiveness(to the needs of people)    
      
Good governance take decisions based on information or knowledge set. Digitization of this information is required for its easy access to all individuals of every community or demography - paving the way for _Digital Governance_. 

Digital governance is a framework for establishing accountability, roles, and decision-making authority for an organization’s digital presence - which means its websites, mobile sites, social channels, and any other Internet and Web-enabled products and services. The intent of digital governance is to ensure that common citizens should be a part of decision-making processes as it affects them directly or indirectly, which in turn improves their conditions and the quality of lives. This new facet of governance will assure that citizens are active contributors in deciding the kind of services they want as well as attentive consumers of services offered to them. And this is what democracy is; _government of the people, by the people and for the people._

How the perspective and participation of people changes with the transformation of governance model from conventional to digital is listed in table 1.

<center><b>Table 1. People Participation in Digital Governance vs. Conventional Governance Models</b></center>
![a](/assets/img/paradigm_shift.png){:class="img-responsive"}

From the matrix above, it is evident that the use of digital governance leads to closer contact of individuals with decision-makers/officials in the government & the impact is immediate. On the whole, it puts greater access and control over governance mechanism in the hands of individuals, and in process leads to a more transparent, accountable and efficient governance. This shift from passive to active to the current need of interactive governance can thus be conceptualized, giving an insight to the social model of governance. 

> _The emergence of social web & the consequential abundant data can be mobilized to define a S-Governance model (where S - stands for Social), a model of government-citizen engagement that complements the web-based e-government services. As a step towards intelligent governance, we expound a new perspective of “Sentiment” in S-governance._

S-governance comprises of two S-factors in the domain of governance : **Social & Sentiment (Sentic)**. The social factor refers to societal interaction of person/entity for their collective co-existence and sentiment describes an expression of strong influence/emotion of people/society. The goal of social and sentiment intelligence based governance is to look forward towards concerned audience and give credence to their views/thoughts for the purpose of information broadcasting, looking for civic inputs in policy making, employment, granting access to services, to uplift and foster stakeholders, etc. The electronic journey of digital governance as the web evolved is represented in figure 2.

![a](/assets/img/web.png){:class="img-responsive"}
<center><b>Figure 2. Evolution of Digital Web Governance</b></center>

---

<center><h1>2. RESEARCH METHODOLOGY</h1></center> 

## 2.1 Systematic Literature Review

A literature survey is carried to review the state-of-the-art research in the area of opinion mining. The studies have been evaluated based on techniques and application areas of opinion mining. As per literature so far, application areas of opinion mining can be broadly classified in to six categories namely, Business Intelligence(BI), Government Intelligence(GI), Information Security & Analysis(ISA), Market Intelligence(MI), Sub Component Technology(SCT) and Smart Society Services(SSS) whereas four classes of techniques have been used so far which include Machine Learning(ML), Lexicon.

The following table 2 represents the summary of opinion mining techniques and the respective application areas in which these have been used as observed from the selected final studies.

<center><b>Table 2. Usage of Opinion Mining Techniques in its application areas</b></center>  
![a](/assets/img/app_area.png){:class="img-responsive"}

Figure 3 and 4 illustrate the percentage of work done in various application areas of opinion mining and the percentage usage of its techniques/approaches in these application areas:

![a](/assets/img/chart1.png){:class="img-responsive"}  
<center><b>Figure 3. Work done in OM Applications (%)</b></center>

![a](/assets/img/chart2.png){:class="img-responsive"}  
<center><b>Figure 4. Work done in OM Techniques (%)</b></center>  

The following charts in figure 5, 6  represent the percentage of work done in various field actions of government intelligence and the percentage use of opinion mining techniques in these action areas.  

![a](/assets/img/chart3.png){:class="img-responsive"}  
<center><b>Figure 5. Work done in Government Intelligence Action Areas (%)</b></center>

![a](/assets/img/chart4.png){:class="img-responsive"}  
<center><b>Figure 6. Work done in GI with OM techniques (%)</b></center>

## 2.2 Research Gaps

Although many new approaches have been proposed, opinion mining remains a challenging task because of the barriers of language, platform, modalities & domain characteristics. Thus, research gaps which have been identified are as follows:

1. Automatically extracting sentiment from texts is difficult as people's expressions of their feelings may be obscure, ambiguous and hard to understand for both human & computer.
2. Analyzing sentiment in user generated contents posted on forums is more difficult than mining regular, formal texts such as news reports. Forum posts are usually short & colloquial, and may contain typos, grammatical errors, forum-specific terms & symbols, and noise such as ads & irrelevant messages.
3. The detection of spam and fake reviews, mainly through the identification of duplicates, the comparison of qualitative with summary reviews, the detection of outliers, and the reputation of the reviewer is an important aspect the limits the results accuracy.
4. Opinion mining must consider domain specific characteristics. For example, in the health domain many sentimental words are used for descriptors of symptoms rather than for expressing personal feelings. A negative sentiment word may not necessarily indicate a negative sentiment. For example, the sentence "When(you're) not feeling well, sweating & shaking, a portable blood glucose meter will help" is actually a neutral, objective expression although negative words are used.
5. Feature extraction being one of the critical task of opinion mining requires optimization.
6. Amongst the various application areas, government intelligence has been identified as the least explored one and requires utmost attention in order to strengthen the socio-economic condition of a nation.
7. The asymmetry in availability of opinion mining software, which can currently be afforded only by organizations and government, but not by citizens. In other words, government have the means today to monitor public opinion in ways that are not available to the average citizens. While content production and publication has democratized, content analysis has not.
8. Multimodality is a challenging task that limits the process of data collection. The representation of communication practices in terms of the textual, aural, linguistic, spatial, and visual resources - or modes - used to compose messages adds an another complexity for opinion mining.
9. The increasing amount of full text material in various languages available over web has increased the difficulty of cross lingual information retrieval process raising another challenge in opinion mining.
10. The continuous need for better usability and user-friendliness of the tools, which are currently usable mainly by data analysts.

---

<center><h1>3. RESEARCH OBJECTIVE</h1></center> 

**_Statement of Research Question_**

> _"Can the pervasive and voluminous user-generated content on social web be mined to gain insights into public opinion for comprehending the paradigm shift from conventional governance to digital governance?"_

In response to the identified need to better exploit the knowledge capital in the form of opinions accumulated on social web, this unifying research question can be broken down into the following three questions, each of which will be addressed by this research:

- How can opinion mining be realized on web 2.0?  
- How can the opinion polarity of user generated big data be determined?  
- How an opinion mining framework for user generated big data facilitate digital governance?  
      
Consequently, the three main research objectives of the work undertaken are:  

1. **Research Objective I** – To seek the convergence of Web 2.0 technologies and opinion mining.  
2. **Research Objective II** – To propose a novel framework for determining opinion polarity of user generated big data.  
3. **Research Objective III** – To find out use case of opinion mining in digital governance by forming a political & social decision support framework for government.  

The objective of this project is to find techniques to automatically determine the opinion of user generated data and gauge the public mood. It specifically aims at developing an intelligent governance model facilitated by the application of opinion mining. This model gain insights in governmental practices and proceedings and provides a generic measure to quantify and qualify the performance of government.

---

<center><h1>4. OVERVIEW OF WORK</h1></center> 

The work encompasses a detailed study of three dimensions of opinion mining namely, techniques, tasks and applications. As per literature, few surveys are there that covers opinion mining evolution, opinion mining techniques, opinion mining task classification, opinion mining and big data etc. but none of them emphasizes over the correlation of these three dimensions. And hence, a systematic literature survey has been done discussing the extent of work done so far in various application areas and techniques of opinion mining. Government intelligence (GI) has been identified as the least explored application area. A very nominal amount of work has been reported in this area and no work to the best of knowledge has been done to gauge public opinion and awareness in government policy initiatives.

Specifically, to improve and optimize decisions and performance of government, the thrust area is to explore and understand opinion mining, its applications and techniques using big data in government intelligence. The idea is to envision “digital governance” by virtue of social web adoption (social media data; source of big data), where government can take advantage of social platforms involving huge user participation.

Thus, the main contribution of this research is two-fold:

> _Firstly, we seek to define a formal unified conceptual model illustrating the relationship of governance with different aspects of policy formation for optimizing the process of government policy cycle by incorporating/adopting the sentic-social aspect. And secondly, we propounds relational frameworks of government in different sectors by offering an optimal predictive learning model based on various machine learning techniques for gauging public opinion on social web._  

In order to define a unified conceptual framework we have considered various case study of the Indian government schemes, policy and programmes and proposed enhanced framework considering various governance aspects. The case study under discussion are as follows: 
      
- **Namami Gange** – One of the national programme launched by the Government of India to address the issues related to conservation and rejuvenation of National river Ganga.
      
- **Demonetization** – This action was directed to limit the shadow economy  and minimize the vitue of black money that was being used to fund corruptive activities and terrorism.
      
- **Budget 2019** - The Interim Union Budget of India for the year 2019 introduced several schemes such as Pradhan Mantri Kisna Samiti.

Further, in order to validate our conceptual framework we seek to build a predictive analytics model based on Opinion Mining. The predictive learning model utilizes various machine learning and deep learning techniques for validating the frameworks. The proposed predictive analytics model comprises of four fundamental statistical tasks, namely data acquisition and preprocessing, feature extraction, design of training model, and metrics based evaluation.

---

<center><h1>5. OPINON MINING MODEL FOR GOVENMENT POLICY EVALUATION</h1></center> 

## 5.1 Three dimensions of Opinion Mining  

The three dimensions of opinion mining are : Techniques, Applications and Task (TAT). Figure 4.1 [32, 233] represents the three dimensions of opinion mining. The specialization of all three dimensions illustrates that all of them are interrelated and can be used in any combination to achieve opinion mining. For example, out of various available techniques {Machine Learning(ML), Lexicon Based (LB), Hybrid Techniques (HT), Ontology Based (OB), Context Based (CB)} any of them could be used to explore opinion/sentiments by performing various tasks (Subjectivity Classification, Sentiment Classification, Review Measures, Spam Detection, Lexicon Formation, Feature Selection and so on) for any available/probable application area ( Government, Market, Business, Smart Society Services, Information Security & Analysis, Sub Component Technology etc).

<center><b>Table 3. Coherence of Three Dimensions with their Specialization</b></center>
![a](/assets/img/relational_matrix.png){:class="img-responsive"}

Table 3 represents the relational matrix of opinion mining three dimensions along with their specialization that has been used in this research work. The selected application area is government where opinion mining has been incorporated in policy evaluation phase of policy life cycle and the proposed opinion mining models have been validated using machine learning and aims to further validate using hybrid techniques (ML + Swarm based, ML + Concept based) by performing tasks such as feature extraction and feature selection.

![a](/assets/img/3d_om.png){:class="img-responsive"}  
<center><b>Figure 7. Three Dimensional Model of Opinion Mining</b></center> 

## 5.2 Techniques Used In Proposed Model  

Various techniques such as machine learning algorithms, lexicon based, hybrid techniques, concept based techniques (contextual and ontology based) etc. have been used in the process of opinion mining so far. Different classifiers have been applied and tested to figure out the best for determining the polarity of an opinion and in the development process of a prediction model. All the aforesaid techniques are based on machine inspired computing that provide machines the ability to learn. The set of combinations of opinion mining techniques that has been used for the validation process of  proposed opinion mining models for governmental policy evaluation are discussed in subsequent sections. A comparative analysis of results based on three parameters of  standard measure of evaluation namely, Precision, Recall and Accuracy has been performed to assess the overall performance of overall opinion polarity classification.

### 5.2.1 Opinion Mining using Machine Learning  

Various techniques such as machine learning algorithms, lexicon based, hybrid techniques, concept based techniques (contextual and ontology based) etc. have been used in the process of opinion mining so far. Different classifiers have been applied and tested to figure out the best for determining the polarity of an opinion and in the development process of a prediction model. All the aforesaid techniques are based on machine inspired computing that provide machines the ability to learn. Various approaches are used for the purpose of opinion polarity classification where machine learning approaches are more popular amongst others. Machine learning techniques helps in making data driven predictions or decisions by the use of various computational methods. The primary objective is to provide the ability of automatic learning to machine(s) without any human intervention This research work empirically  analyze  certain standard machine learning algorithms listed in table 4. 

<center><b>Table 4. Machine Learning Techniques Used</b></center>
![a](/assets/img/ml_algo.png){:class="img-responsive"}  

### 5.2.2 Opinion Mining using Hybrid Techniques  

Various literature reveals the use of hybrid approach for opinion mining. Hybrid Approaches are the combination of different opinion mining  techniques collaborated for a better performance. We aim to incorporate two such hybrid approaches as a part of further extension, Swarm based and Ontology driven.

## 5.3 Tasks performed in Proposed model  

Feature extraction (FE) is one of the critical and complex tasks in opinion mining. The objective is to recognize the entity (person, service or an object) that is being referred in opinion. It is one of the significant step in the process of polarity classification (opinion polarity: positive, negative, neutral) that converts input data (unstructured textual opinion indicated data), into an array of representative features. Commonly, the feature extraction task is done using intrinsic ‘filtering’ methods which are fast, classifier-independent methods that rank features according to predetermined numerical functions based on the measure of the “importance” of the terms. Various scoring functions namely, cross-entropy, chi-square, mutual information, TF-IDF, information gain, etc. can be used to pick features with the highest scores for statistical measures. The strength of classifier's accuracy is directly proportional to the selected quality data features (i.e. training dataset) that are typically prepared manually. Literature reveals that an optimal process of feature selection enhances the performance of classifier (over the parameters of predictive power, speed and model simplicity), minimize noise, reduces dimensionality, and helps in data visualization for model selection. In feature selection the features are kept intact and n best features are chosen among them, removing the redundant and co-linear features. There are numerous computational  and financial challenges associated with task of relevant feature selection and discarding the non-essential. Motivated by these issues and in order to optimize the feature space without sacrificing remarkable classification accuracy, two intelligent data analytics solution for opinion prediction has been used and compared with the conventional feature extraction. The details of conventional and optimal feature extraction are as follows:

### **a) Conventional Feature Extraction (TF-IDF Based)**  
TF-IDF stands for Term Frequency - Inverse Document Frequency. It is a weight statistically measured to evaluate the importance of a word to a document in a corpus. The importance of a word increases as its frequency increases in a document but is offset by the frequency of word in corpus. The Term Frequency, TF(t, d) simply counts the frequetemplatesncy of a term in a document as follows: 

### **b) Swarm optimized Feature Extraction (Binary Bat based TF-IDF)**  
Feature extraction or selection is an important task carried out in the process of opinion mining. The purpose of feature selection is to (a) refrain from the curse of dimensionality (b) enhance generalization by reducing over fitting (c) simplify models for easier interpretation by researchers and (d) make shorter training times. The task is all about to choose a subset of related features from the initial original feature set. As a next step, we seek to employ one of the widely used swarm technique i.e. the binary bat search algorithm developed by R.Y.M Nakamura et al. (2012). It will  be employed for enhanced opinion classification using optimized feature selection.

### **c) Concept based Feature Extraction (Ontology driven TF-IDF)**   
Ontology is specifically defined as a conceptual reference model that describes the semantics of a system or domain. It represents the relationship between concepts; both in human comprehensible and machine processable manner. It represents a concept or categories of a particular subject area that exhibits the characteristics and relationship between them.

## 5.4 Opinion Mining Model  

![a](/assets/img/ml_model.png){:class="img-responsive"}  
<center><b>Figure 8. Opinion Mining Model</b></center>  

Data acquisition is the preliminary step and involve the use of data extraction techniques from social media platform “Twitter” using twitter search API. Preprocessing is the further step which involves nomalization of the extracted text by performing stemming, lemmatization and futher necessary steps.

Various combination of techniques is used for the design of training model which involves the use of traditional machine learning algorithms like Logistic Regression, Support Vector Machine, Random Forest, XGBoost and Multi Layer Perceptron. Some other model is built by embedding the concept of deep learning model like Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) or some form of it like LSTM/GRU. Few model was experimented by developing a sentiment analysis model using N-gram Multichannel Convolutional Neural Network.

As a final step we evaluated the fit model by predicting the sentiment on all reviews in the unseen test dataset and the performance has been compared over results. Precision, Recall, Accuracy and F1-score have been used as a performance measures to evaluate the efficacy of the classifier. 

---

<center><h1>6. Application Area in Proposed Model</h1></center>  

The application areas of opinion mining model includes business intelligence, information security & analysis, government intelligence, sub component technology, market intelligence and smart society services. Amongst all, government domain has been selected in this research work being the least explored area of opinion mining as per literature review. Several fields of actions of government domain are politics, voting advisory, policy making, advocacy monitoring, policy evaluation, campaigning, detection of early warning system, conservation of natural resources etc. The policy life cycle development model has been studied and the importance of incorporating the concept of opinion mining is realized. Various opinion mining models have been proposed for governmental policy evaluation and has been validated over discussed opinion mining techniques. Different case studies used for this purpose has been discussed.

---

> Disclaimer: This is a working article, and hence it represents research in progress. This article represents the opinions of the authors, and is the product of professional research. This article is provided for information only.

---

Image by <a href="https://pixabay.com/users/wynpnt-868761/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2296821">Wynn Pointaux</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2296821">Pixabay</a>

> _In case if you found something useful to add to this article or would like to improve some points mentioned, feel free to write it down in the comments. Hope you found something useful here._
