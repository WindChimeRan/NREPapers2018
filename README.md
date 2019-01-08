# Relation Extraction in 2018

This repo covers almost all the papers (35) related to **Neural Relation Extraction** in ACL, EMNLP, COLING, NAACL, AAAI, IJCAI in 2018.

Use tags to search papers you like.

**tags: Multi-lingual NRE | DSRE | new task | NRC | Joint extraction of both entity and relation | bootstrapping RE | few shot | GCN | capsule network**

DSRE: Distant Supervision Relation Extraction

NRE: Neural Relation Extraction

NRC: Neural Relation Classification

GCN: Graph Convolution Network

Note that DSRE is a big track in RE, including _a denoising preprocessing step_ | _bootstrapping RE_ | _Learning in the noise_ .etc

## COLING 2018

1. **Adversarial Multi-lingual Neural Relation Extraction**
    _Xiaozhi Wang, Xu Han, Yankai Lin, Zhiyuan Liu and Maosong Sun._
    COLING 2018
    [paper](http://aclweb.org/anthology/C18-1099) [code](https://github.com/thunlp/AMNRE)
    
    Multi-lingual NRE
    >Existing models cannot well capture the consistency and diversity of relation patterns in different languages. To address these issues, we propose an adversarial multi-lingual neural relation extraction (AMNRE) model, which builds both consistent and individual representations for each sentence to consider the consistency and diversity among languages. Further, we adopt an adversarial training strategy to ensure those consistent sentence representations could effectively extract the language-consistent relation patterns.

2. **Cooperative Denoising for Distantly Supervised Relation Extraction**
    _Kai Lei, Daoyuan Chen, Yaliang Li, Nan Du, Min Yang, Wei Fan and Ying Shen._
    COLING 2018
    [paper](http://aclweb.org/anthology/C18-1036)

    DSRE: Learning in the noise.
    >we propose a novel neural relation extraction framework with bi-directional knowledge distillation to cooperatively use different information sources and alleviate the noisy label problem in distantly supervised relation extraction.

3. **Exploratory Neural Relation Classification for Domain Knowledge Acquisition**
    _Yan Fan, Chengyu Wang and Xiaofeng He._
    COLING 2018
    [paper](http://aclweb.org/anthology/C18-1192)

    Exploratory Relation Classification (ERC): **NRC + new task**!
    >In this paper, we propose the task of ERC to address the problem of domain-specific knowledge acquisition. We propose a DSNN model to address the task, consisting of three modules, an integrated base neural network for relation classification, a similarity-based clustering algorithm ssCRP to generate new relations and constrained relation prediction process with the purpose of populating new relations.

4. **Multilevel Heuristics for Rationale-Based Entity Relation Classification in Sentences**
    _Shiou Tian Hsu, Mandar Chaudhary and Nagiza Samatova._
    COLING 2018
    [paper](http://aclweb.org/anthology/C18-1098)

    NRC + rationale interpretability **NRC + new task**!

    >In this paper, we have proposed an improved rationale-based model for entity relation classification. In our model, besides context word information, we also moderate rationale generation with multiple heuristics computed from different text level features. 

5. **Neural Relation Classification with Text Descriptions**
    _Feiliang Ren, Di Zhou, Zhihui Liu, Yongcheng Li, Rongsheng Zhao, Yongkang Liu and Xiaobo Liang._
    COLING 2018
    [paper](http://aclweb.org/anthology/C18-1100)

    NRC

    >In this paper, we propose DesRC, a new neural relation classification method which integrates entities text descriptions into deep neural networks models. We design a two-level attention mechanism to select the most useful information from the ”intra-sentence” aspect and the ”cross-sentence” aspect. Besides, the adversarial training method is also used to further improve the classification performance.


6. **Adversarial Feature Adaptation for Cross-lingual Relation Classification**
    _Bowei Zou, Zengzhuang Xu, Yu Hong and Guodong Zhou._
   COLING 2018
   [paper](http://aclweb.org/anthology/C18-1037)
   
    Multi-lingual NRE
    
    > In this paper, we come up with a feature adaptation approach for cross-lingual relation classification, which employs a generative adversarial network (GAN) to transfer feature representations from one language with rich annotated data to another language with scarce annotated data. 

## ACL 2018

1. **Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism.**
    _Xiangrong Zeng, Daojian Zeng, Shizhu He, Kang Liu, Jun Zhao._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-1047)
    
    Joint extraction of both entity and relation
    
    >In this paper, we propose an end-to-end model based on sequence-to-sequence learning with copy mechanism, which can jointly extract relational facts from sentences of any of these classes. 

2. **Adaptive Scaling for Sparse Detection in Information Extraction.**
    _Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-1095)
    
    >In this paper, we propose adaptive scaling, an algorithm which can handle the positive sparsity problem and directly optimize over F-measure via dynamic costsensitive learning.
    
3. **Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning.** 
    _Pengda Qin, Weiran XU, William Yang Wang._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-1199)
    
    DSRE: a denoising preprocessing step
    
    >We explore a deep reinforcement learning strategy to generate the false-positive indicator, where we automatically recognize false positives for each relation type without any supervised information. Unlike the removal operation in the previous studies, we redistribute them into the negative examples.
    
4. **DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction.**
    _Pengda Qin, Weiran XU, William Yang Wang._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-1046)
    
    DSRE: a denoising preprocessing step
    
    >Inspired by Generative Adversarial Networks, we regard the positive samples generated by the generator as the negative samples to train the discriminator. The optimal generator is obtained until the discrimination ability of the discriminator has the greatest decline. We adopt the generator to filter distant supervision training dataset and redistribute the false positive instances into the negative set, in which way to provide a cleaned dataset for relation classification.

5. **A Walk-based Model on Entity Graphs for Relation Extraction.**
    _Fenia Christopoulou, Makoto Miwa, Sophia Ananiadou._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-2014)
    
    NRC
    
    >We present a novel graph-based neural network model for relation extraction. Our model treats multiple pairs in a sentence simultaneously and considers interactions among them. All the entities in a sentence are placed as nodes in a fully-connected graph structure.
    
6. **Ranking-Based Automatic Seed Selection and Noise Reduction for Weakly Supervised Relation Extraction.**
    _Van-Thuy Phi, Joan Santoso, Masashi Shimbo, Yuji Matsumoto._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-2015)

    DSRE | bootstrapping RE
    
    >This paper addresses the tasks of automatic seed selection for bootstrapping relation extraction, and noise reduction for distantly supervised relation extraction.

## AAAI 2018

1. **Large Scaled Relation Extraction with Reinforcement Learning**
    _Xiangrong Zeng, Shizhu He, Kang Liu, Jun Zhao_
    AAAI 2018
    [paper](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/zeng_aaai2018.pdf)
    
    DSRE: sentence level rather than bag level
    
    >In this paper, we learn the relation extractor with reinforcement learning method on the distant supervised dataset. The bag relation is used as the distant supervision which guide the training of relation extractor. We also apply the relation extractor to help bag relation extraction

2. **SEE: Syntax-aware Entity Embedding for Neural Relation Extraction**
    _Zhengqiu He*, Wenliang CHEN, Meishan Zhang, Zhenghua Li, Wei Zhang, Min Zhang_
    AAAI 2018
    [paper](http://arxiv.org/abs/1801.03603)
    
    DSRE
    
    >we propose to learn syntax-aware entity embedding for neural relation extraction. First, we encode the context of entities on a dependency tree as sentence-level entity embedding based on tree-GRU. Then, we utilize both intra-sentence and inter-sentence attentions to obtain sentence set-level entity embedding over all sentences containing the focus entity pair. Finally, we combine both sentence embedding and entity embedding for relation classification.

3. **Reinforcement Learning for Relation Classification from Noisy Data**
    _Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu_
    AAAI2018
    [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/AAAI2018Denoising.pdf)
    
    DSRE: sentence level rather than bag level
    >we propose a novel model for relation classification at the sentence level from noisy data. The model has two modules: an instance selector and a relation classifier. The instance selector chooses high-quality sentences with learning and feeds the selected sentences into the relation classifier, and the relation classifier makes sentencelevel prediction and provides rewards to the instance selector. The two modules are trained jointly to optimize the instance selection and relation classification processes.

## IJCAI 2018

1. **Joint Extraction of Entities and Relations Based on a Novel Graph Scheme**
    _Shaolei Wang, Yue Zhang, Wanxiang Che, Ting Liu_
    IJCAI 2018
    [paper](https://www.ijcai.org/proceedings/2018/0620.pdf)
    
    Joint extraction of both entity and relation
    
    > In this paper, we convert the joint task into a directed graph by designing a novel graph scheme and propose a transition-based approach to generate the directed graph incrementally, which can achieve joint learning through joint decoding. Our method can model underlying dependencies not only between entities and relations, but also between relations.
    
2. **Ensemble Neural Relation Extraction with Adaptive Boosting**
    _Dongdong Yang, Senzhang Wang, Zhoujun Li_
    IJCAI 2018
    [paper](https://www.ijcai.org/proceedings/2018/0630.pdf)
    
    DSRE
    
    >we proposed to integrate attention-based LSTMs with adaptive boosting model for relation extraction.

3. **Exploring Encoder-Decoder Model for Distant Supervised Relation Extraction**
    _Sen Su, Ningning Jia, Xiang Cheng, Shuguang Zhu, Ruiping Li_
    IJCAI 2018
    [paper](https://www.ijcai.org/proceedings/2018/0610.pdf)
    
    DSRE
    
    >we present a simple yet effective encoderdecoder model for distant supervised relation extraction. Given the sentence bag of an entity pair as input, the CNN encoder extracts sentence features and merge them into a bag representation. While the LSTM decoder leverages the dependencies among the relations by predicting them in a sequential manner. To enable the sequential prediction of relations, we introduce a measure to quantify the amounts of information contained in a sentence bag for its relations, which are used to determine relation orders during training to let the model predict relations in a descending order of their amounts of information.

## NAACL 2018

1. **GLOBAL RELATION EMBEDDING FOR RELATION EXTRACTION**
    _Yu Su, Honglei Liu, Semih Yavuz, Izzeddin Gur, Huan Sun and Xifeng Yan_
    NAACL 2018
    [paper](http://aclweb.org/anthology/N18-1075)
    
    DSRE
    >we propose to embed textual relations with global statistics of relations, i.e., the cooccurrence statistics of textual and knowledge base relations collected from the entire corpus.

2. **JOINT BOOTSTRAPPING MACHINES FOR RELATION EXTRACTION**
    _Pankaj Gupta and Hinrich Schütze_
    NAACL 2018
    [paper](http://aclweb.org/anthology/N18-1003)
    
    bootstrapping RE
    
    >We have proposed a Joint Bootstrapping Machine for relation extraction (BREJ) that takes advantage of both entity-pair-centered and template-centered approaches. We have demonstrated that the joint approach scales up positive instances that boosts the confidence of NNLC extractors and improves recall. 

3. **SIMULTANEOUSLY SELF-ATTENDING TO ALL MENTIONS FOR FULL-ABSTRACT BIOLOGICAL RELATION EXTRACTION**
    _Patrick Verga, Emma Strubell and Andrew McCallum_
    NAACL 2018
    [paper](http://aclweb.org/anthology/N18-1080)
    
   **document-level RE + new task** 
   >We present a bi-affine relation attention network that simultaneously scores all mention pairs within a document.
   
   >**a new, large and high-quality dataset introduced in this work.**

4. **STRUCTURE REGULARIZED NEURAL NETWORK FOR ENTITY RELATION CLASSIFICATION FOR CHINESE LITERATURE TEXT**
    _Ji Wen, Xuancheng Ren, Xu Sun and Qi Su_
    NAACL 2018
    [paper](http://aclweb.org/anthology/N18-2059)
    
    **Chinese NRC + new task**
    >We present a novel model, named Structure Regularized Bidirectional Recurrent Convolutional Neural Network (SR-BRCNN), to identify the relation between entities. The proposed model learns relation representations along the shortest dependency path (SDP) extracted from the structure regularized dependency tree, which has the benefits of reducing the complexity of the whole model.

## EMNLP 2018

1. **Multi-Level Structured Self-Attentions for Distantly Supervised Relation Extraction**
    _Jinhua Du, Jingguang Han, Andy Way and Dadong Wan_
    EMNLP 2018
    [paper](https://arxiv.org/pdf/1809.00699.pdf)
    
    DSRE
    
    >This paper has proposed a multi-level structured self-attention mechanism for distantly supervised RE. In this framework, the traditional 1-D wordlevel and sentence-level attentions are extended to 2-D structured matrices which can learn different aspects of a sentence, and different informative instances.
    
2. **Hierarchical Relation Extraction with Coarse-to-Fine Grained Attention**
    _Xu Han, Pengfei Yu, Zhiyuan Liu, Maosong Sun and Peng Li_
    EMNLP 2018
    
    paper not released
    
    [ppt](https://www.leiphone.com/news/201809/FZM8dsm5CcQpmnNT.html)

3. **Neural Relation Extraction via Inner-Sentence Noise Reduction and Transfer Learning**
    _Tianyi Liu, Xinsong Zhang, Wanhao Zhou and Weijia Jia_
    EMNLP 2018
    [paper](https://arxiv.org/pdf/1808.06738.pdf)
    
    DSRE
    
    >We first build Sub-Tree Parse (STP) to remove noisy words that are irrelevant to relations. Then we construct a neural network inputting the subtree while applying the entity-wise attention to identify the important semantic features of relational words in each instance. To make our model more robust against noisy words, we initialize our network with a priori knowledge learned from the relevant task of entity classification by transfer learning. 

4. **N-ary Relation Extraction using Graph State LSTM**
    _Linfeng Song, Yue Zhang, Zhiguo Wang and Daniel Gildea_
    EMNLP 2018
    [paper](https://arxiv.org/pdf/1808.09101.pdf)

    N-ary relation extraction | GCN related
    
    >We propose a graph-state LSTM model, which uses a parallel state to model each word, recurrently enriching state values via message passing.
    
5. **RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information**
    _Shikhar Vashishth, Rishabh Joshi, Sai Suman Prayaga, Chiranjib Bhattacharyya and Partha Talukdar_
    EMNLP 2018
    [paper](http://malllabiisc.github.io/publications/papers/reside_emnlp18.pdf)
    
    DSRE | GCN
    
    >utilizes additional side information from KBs for improved relation extraction. It uses entity type and relation alias information for imposing soft constraints while predicting relations. RESIDE employs Graph Convolution Networks (GCN) to encode syntactic information from text and improves performance even when limited side information is available.

6. **Label-Free Distant Supervision for Relation Extraction via Knowledge Graph Embedding**
    _Guanying Wang, Wen Zhang, Ruoxu Wang, Yalin Zhou and Huajun Chen_
    EMNLP 2018
    
    [paper](http://www.aclweb.org/anthology/D18-1248)
    
    DSRE | KGE
    
    [zhihu](https://zhuanlan.zhihu.com/p/43326102)
    
    > What if TransE + PCNN? What if we replace entity mentions with entity types? Learn Label-Free DSRE!
    
7. **Graph Convolution over Pruned Dependency Trees Improves Relation Extraction**
    _Yuhao Zhang, Peng Qi and Christopher D. Manning_
    EMNLP 2018
    
    [paper](https://nlp.stanford.edu/pubs/zhang2018graph.pdf)
    
    GCN
    
    >We propose an extension of graph convolutional networks that is tailored for relation extraction, which pools information over arbitrary dependency structures efficiently in parallel. To incorporate relevant information while maximally removing irrelevant content, we further apply a novel pruning strategy to the input trees by keeping words immediately around the shortest path between the two entities among which a relation might hold.

8. **Extracting Entities and Relations with Joint Minimum Risk Training**
    _sun changzhi, Yuanbin Wu and Man Lan_
    
    EMNLP 2018
    
    paper not released

9. **Large-scale Exploration of Neural Relation Classification Architectures**
    _Hoang-Quynh Le, Duy-Cat Can, Sinh T. Vu, Thanh Hai Dang, Mohammad Taher Pilehvar and Nigel Collier_
    
    EMNLP 2018
    
    paper not released
    
10. **Adversarial training for multi-context joint entity and relation extraction**
    _Giannis Bekoulis, Johannes Deleu, Thomas Demeester and Chris Develder_
    
    EMNLP 2018

    [paper](https://arxiv.org/pdf/1808.06876.pdf)
    
    Joint extraction of both entity and relation
    
    (related paper: [Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/pdf/1804.07847.pdf))
    
    >we demonstrate that applying AT to a general purpose baseline model for jointly extracting entities and relations, allows improving the state-of-the-art effectiveness on several datasets.
    
11. **FewRel: A Large-Scale Supervised Few-shot Relation Classification Dataset with State-of-the-Art Evaluation**
    _Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu and Maosong Sun_
    
    EMNLP 2018
    
    few-shot
    
    paper not released

    [ppt](https://www.leiphone.com/news/201809/FZM8dsm5CcQpmnNT.html)

12. **Genre Separation Network with Adversarial Training for Cross-genre Relation Extraction**
    _Ge Shi, Chong Feng, Lifu Huang, Boliang Zhang, Heng Ji, Lejian Liao and Heyan Huang_
    
    EMNLP 2018
    
    [paper](http://nlp.cs.rpi.edu/paper/relation2018.pdf)
    
    NRC | cross-genre relation extraction
    
    >we design a genre-separation network, which applies two encoders, one genre independent and one genre-shared, to explicitly extract genre-specific and genre-agnostic features. Then we train a relation classifier using the genre-agnostic features on the source genre and directly apply to the target genre.

13. **Attention-Based Capsule Network with Dynamic Routing for Relation Extraction**
    _Ningyu Zhang, Shumin Deng, Huajun Chen, Zhanling Sun, Yiyi Zhang and Xiaoqian Li_
    
    EMNLP 2018
    
    [paper](http://aclweb.org/anthology/D18-1120)
    
    capsule network
    
    > Capsule Network is good at multi-label classification in RE in NLP!
    
