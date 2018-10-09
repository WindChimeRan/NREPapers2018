# Relation Extraction in 2018

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

6. **Word-Level Loss Extensions for Neural Temporal Relation Classification**
    _Artuur Leeuwenberg and Marie-Francine Moens._
    COLING 2018
    [paper](http://aclweb.org/anthology/C18-1291)

    Temporal Relation Classification

    >In this work, we extend our classification model’s task loss with an unsupervised auxiliary loss on the word-embedding level of the model. 

7. **Adversarial Feature Adaptation for Cross-lingual Relation Classification**
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
    
    
    >We present a novel graph-based neural network model for relation extraction. Our model treats multiple pairs in a sentence simultaneously and considers interactions among them. All the entities in a sentence are placed as nodes in a fully-connected graph structure.
    
6. **Ranking-Based Automatic Seed Selection and Noise Reduction for Weakly Supervised Relation Extraction.**
    _Van-Thuy Phi, Joan Santoso, Masashi Shimbo, Yuji Matsumoto._
    ACL 2018
    [paper](http://aclweb.org/anthology/P18-2015)

    DSRE|bootstrapping RE
    >This paper addresses the tasks of automatic seed selection for bootstrapping relation extraction, and noise reduction for distantly supervised relation extraction.

## AAAI 2018
