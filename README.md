## Resource List

个人整理的人工智能、区块链的资源

---

### 一、AI项目

#### ONNX

https://github.com/onnx/onnx

Open Neural Network Exchange (ONNX) is the first step toward an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Initially we focus on the capabilities needed for inferencing (evaluation).

Caffe2, PyTorch, Microsoft Cognitive Toolkit, Apache MXNet and other tools are developing ONNX support. Enabling interoperability between different frameworks and streamlining the path from research to production will increase the speed of innovation in the AI community. We are an early stage and we invite the community to submit feedback and help us further evolve ONNX.

#### MXNet

https://github.com/apache/incubator-mxnet

Apache MXNet (incubating) is a deep learning framework designed for both efficiency and flexibility. It allows you to mix symbolic and imperative programming to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of blue prints and guidelines for building deep learning systems, and interesting insights of DL systems for hackers.

#### MXNet Model Server

https://github.com/awslabs/mxnet-model-server

Apache MXNet Model Server (MMS) is a flexible and easy to use tool for serving deep learning models exported from MXNet or the Open Neural Network Exchange (ONNX).

Use the MMS Server CLI, or the pre-configured Docker images, to start a service that sets up HTTP endpoints to handle model inference requests.

#### Lucid

https://github.com/tensorflow/lucid

Lucid is a collection of infrastructure and tools for research in neural network interpretability.

#### Sonnet

https://github.com/deepmind/sonnet

Sonnet is a library built on top of TensorFlow for building complex neural networks.

#### The DeepMind Control Suite and Package

https://github.com/deepmind/dm_control

This package contains:
- A set of Python Reinforcement Learning environments powered by the MuJoCo physics engine. See the suite subdirectory.
- Libraries that provide Python bindings to the MuJoCo physics engine.

#### CNTK

https://github.com/Microsoft/CNTK

Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit

#### Microsoft Machine Learning for Apache Spark

https://github.com/Azure/mmlspark

MMLSpark provides a number of deep learning and data science tools for Apache Spark, including seamless integration of Spark Machine Learning pipelines with Microsoft Cognitive Toolkit (CNTK) and OpenCV, enabling you to quickly create powerful, highly-scalable predictive and analytical models for large image and text datasets.

#### Microsoft - MMdnn

https://github.com/Microsoft/MMdnn

MMdnn is a set of tools to help users inter-operate among different deep learning frameworks. E.g. model conversion and visualization. Convert models between Caffe, Keras, MXNet, Tensorflow, CNTK, PyTorch and CoreML.

#### BigDL

https://github.com/intel-analytics/BigDL

BigDL: Distributed Deep Learning on Apache Spark

#### edward

https://github.com/blei-lab/edward

Edward is a Python library for probabilistic modeling, inference, and criticism. It is a testbed for fast experimentation and research with probabilistic models, ranging from classical hierarchical models on small data sets to complex deep probabilistic models on large data sets. Edward fuses three fields: Bayesian statistics and machine learning, deep learning, and probabilistic programming.

#### Python Codes in Data Science

https://github.com/RubensZimbres/Repo-2017

Python codes in Machine Learning, NLP, Deep Learning and Reinforcement Learning with Keras and Theano

#### Featuretools

https://github.com/Featuretools/featuretools

Featuretools is a python library for automated feature engineering.

#### Keras中文文档

https://keras.io/zh/

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。

如果你有如下需求，请选择 Keras：
- 允许简单而快速的原型设计（用户友好，高度模块化，可扩展性）。
- 同时支持卷积神经网络和循环神经网络，以及两者的组合。
- 在CPU和GPU上无缝运行与切换。

---

### 二、NLP项目

#### Tracking Progress in Natural Language Processing

https://github.com/sebastianruder/NLP-progress

This document aims to track the progress in Natural Language Processing (NLP) and give an overview of the state-of-the-art across the most common NLP tasks and their corresponding datasets.

#### baseline

https://github.com/dpressel/baseline

Simple, Strong Deep-Learning Baselines for NLP in several frameworks

Baseline algorithms and data support implemented with multiple deep learning tools, including sentence classification, tagging, seq2seq, and language modeling. Can be used as stand-alone command line tools or as a Python library. The library attempts to provide a common interface for several common deep learning tasks, as well as easy-to-use file loaders to make it easy to publish standard results, compare against strong baselines without concern for mistakes and to support rapid experiments to try and beat these baselines.

#### Natural Language Toolkit (NLTK)

https://github.com/nltk/nltk

NLTK -- the Natural Language Toolkit -- is a suite of open source Python modules, data sets, and tutorials supporting research and development in Natural Language Processing.

#### ParlAI

https://github.com/facebookresearch/ParlAI

ParlAI (pronounced “par-lay”) is a framework for dialog AI research, implemented in Python.

Its goal is to provide researchers:
- a unified framework for sharing, training and testing dialog models
- many popular datasets available all in one place, with the ability to multi-task over them
- seamless integration of Amazon Mechanical Turk for data collection and human evaluation

#### DeepQA

https://github.com/allenai/deep_qa

DeepQA is a library for doing high-level NLP tasks with deep learning, particularly focused on various kinds of question answering. DeepQA is built on top of Keras and TensorFlow, and can be thought of as an interface to these systems that makes NLP easier.

#### Deep - NLP

https://github.com/siddk/deep-nlp

This repository contains Tensorflow implementations of various deep learning models, with a focus on problems in Natural Language Processing. Each individual subdirectory is self-contained, addressing one specific model.

#### 清华大学中文自然文本数据集CTW

https://ctwdataset.github.io/

we provide details of a newly created dataset of Chinese text with about 1 million Chinese characters annotated by experts in over 30 thousand street view images. This is a challenging dataset with good diversity. It contains planar text, raised text, text in cities, text in rural areas, text under poor illumination, distant text, partially occluded text, etc. For each character in the dataset, the annotation includes its underlying character, its bounding box, and 6 attributes. The attributes indicate whether it has complex background, whether it is raised, whether it is handwritten or printed, etc.

#### Kaldi Speech Recognition Toolkit

https://github.com/tramphero/kaldi

This is now the official location of the Kaldi project. http://kaldi-asr.org

#### NLP Architect by Intel® AI LAB

https://github.com/NervanaSystems/nlp-architect

NLP Architect by Intel AI Lab: Python library for exploring the state-of-the-art deep learning topologies and techniques for natural language processing and natural language understanding http://nlp_architect.nervanasys.com/

$\alpha$

---

### 三、学习资源

#### Oxford NLP Lectures

https://github.com/oxford-cs-deepnlp-2017/lectures

This repository contains the lecture slides and course description for the Deep Natural Language Processing course offered in Hilary Term 2017 at the University of Oxford.

#### stat212b

https://github.com/joanbruna/stat212b

Topics Course on Deep Learning for Spring 2016, by Joan Bruna, UC Berkeley, Statistics Department

#### Deep-Learning-101

https://github.com/sjchoi86/Deep-Learning-101

Deep Learning Tutorials These tutorials are for deep learning beginners which have been used in a six week Deep Learning and Computer Vision course. Hope these to be helpful for understanding what deep learning is and how it can be applied to various fields including computer vision, robotics, natural language processings, and so forth.

#### Stanford Machine Learning course exercises

https://github.com/krasserm/machine-learning-notebooks

Stanford Machine Learning course exercises implemented with scikit-learn

#### CMU 10703: Deep Reinforcement Learning and Control, Spring 2017

https://katefvision.github.io/

+ Implement and experiment with existing algorithms for learning control policies guided by reinforcement, expert demonstrations or self-trials.
+ Evaluate the sample complexity, generalization and generality of these algorithms.
+ Be able to understand research papers in the field of robotic learning.
+ Try out some ideas/extensions of your own. Particular focus on incorporating true sensory signal from vision or tactile sensing, and exploring the synergy between learning from simulation versus learning from real experience.

#### MIT 6.S099: Artificial General Intelligence

https://agi.mit.edu/

This class takes an engineering approach to exploring possible research paths toward building human-level intelligence. The lectures will introduce our current understanding of computational intelligence and ways in which strong AI could possibly be achieved, with insights from deep learning, reinforcement learning, computational neuroscience, robotics, cognitive modeling, psychology, and more. Additional topics will include AI safety and ethics. Projects will seek to build intuition about the limitations of state-of-the-art machine learning approaches and how those limitations may be overcome. The course will include several guest talks. Listeners are welcome.

#### The Human Brain

https://nancysbraintalks.mit.edu/course/9-11-the-human-brain

MIT认知神经科学教授Nancy Kanwisher，放出了一大波本学期（2018年春季）MIT本科生课程人类大脑（The Human Brain）的视频，课程代号MIT 9.11。

#### Berkeley 人工智能相关课程

http://bair.berkeley.edu/courses.html

#### Berkeley CS 294: Deep Reinforcement Learning, Fall 2017

http://rll.berkeley.edu/deeprlcourse/#syllabus

#### Tensorflow-101

https://github.com/sjchoi86/Tensorflow-101

Tensorflow Tutorials using Jupyter Notebook

TensorFlow tutorials written in Python (of course) with Jupyter Notebook. Tried to explain as kindly as possible, as these tutorials are intended for TensorFlow beginners. Hope these tutorials to be a useful recipe book for your deep learning projects. Enjoy coding! :)

#### Deep Learning Papers Reading Roadmap

https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap

If you are a newcomer to the Deep Learning area, the first question you may have is "Which paper should I start reading from?"

Here is a reading roadmap of Deep Learning papers!

#### 深度学习论文整理

https://github.com/terryum/awesome-deep-learning-papers

Awesome - Most Cited Deep Learning Papers

A curated list of the most cited deep learning papers (since 2012)

We believe that there exist classic deep learning papers which are worth reading regardless of their application domain. Rather than providing overwhelming amount of papers, We would like to provide a curated list of the awesome deep learning papers which are considered as must-reads in certain research domains.

#### 谷歌机器学习速成课25讲

https://developers.google.cn/machine-learning/crash-course/

课程的主要架构分为三个部分：机器学习概念（18讲）、机器学习工程（4讲）、机器学习在现实世界的应用示例（3讲）。

#### 乔治亚理工大学自然语言处理教材

https://github.com/jacobeisenstein/gt-nlp-class/tree/master/notes

These notes are the basis for the readings in CS4650 and CS7650 ("Natural Language") at Georgia Tech.

乔治亚理工大学 Jacob Eisenstein 教授开放了自然语言处理领域的最新教材《Natural Language Processing》

---

### 四、区块链项目

#### Hyperledger

https://github.com/hyperledger/hyperledger

Hyperledger is an open source collaborative effort created to advance cross-industry blockchain technologies. It is a global collaboration including leaders in finance, banking, Internet of Things, supply chains, manufacturing and Technology. The Linux Foundation hosts Hyperledger under the foundation.

#### 区块链技术指南

https://github.com/yeasy/blockchain_guide

本书希望可以探索区块链概念的来龙去脉，剥茧抽丝，剖析关键技术原理，同时讲解实践应用。在开发相关开源分布式账本平台（如超级账本），以及设计基于区块链的企业方案过程中，笔者积累了一些实践经验，也通过本书一并分享出来，希望能推动区块链技术的早日成熟和更多应用场景的出现。

#### EOS项目

https://github.com/EOSIO/eos


---

### 五、量化交易

#### awesome-quant

https://github.com/wilsonfreitas/awesome-quant

A curated list of insanely awesome libraries, packages and resources for Quants (Quantitative Finance)

#### Kungfu Master Trading System

https://github.com/taurusai/kungfu

功夫是 Taurus.ai 团队专为量化交易者设计的低延迟交易执行系统。功夫想要解决两个核心问题：
- 交易数据实时存储 - 为了追求交易速度，很多情况下交易者不得不放弃存储功能以换取更高的执行速度，这给盘后分析带来很大困难，使分析缺乏必要的数据支持。
- 交易延迟分析及优化 - 交易者缺乏测量交易延迟的标准化工具，关于延迟的统计数据往往只能凭感觉估算，在缺乏可靠的数据采集手段的情况下，优化工作变得更加艰难。

#### vnpy

https://github.com/vnpy/vnpy

vn.py是基于Python的开源量化交易程序开发框架，起源于国内私募的自主量化交易系统。2015年初项目启动时只是单纯的交易API接口的Python封装。随着业内关注度的上升和社区不断的贡献，目前已经成长为一套全功能的交易程序开发框架，用户群体也日渐多样化，包括私募基金、券商自营和资管、期货资管和子公司、高校研究机构和专业个人投资者等。

#### Deep-Trading

https://github.com/Rachnog/Deep-Trading

Algorithmic trading with deep learning experiments. Now released part one - simple time series forecasting. I plan to implement more sophisticated algorithms and their ensembles with different features, check their performance, train a trading strategy and go live.

#### stock market reinforcement learning

https://github.com/kh-kim/stock_market_reinforcement_learning

This project provides a general environment for stock market trading simulation using OpenAI Gym. Training data is a close price of each day, which is downloaded from Google Finance, but you can apply any data if you want. Also, it contains simple Deep Q-learning and Policy Gradient from Karpathy's post. http://karpathy.github.io/2016/05/31/rl/

---

### 六、数学资源

#### Awesome Math

https://github.com/rossant/awesome-math

A curated list of awesome mathematics resources.

#### 一个latex模版

https://github.com/goodfeli/dlbook_notation

---

### 七、自动机器学习资源

#### auto-sklearn

https://github.com/automl

https://github.com/automl/auto-sklearn

#### Machine Learning for .NET

https://github.com/dotnet/machinelearning

主要学习框架

#### MLPerf Reference

https://github.com/mlperf/reference

Reference implementations of MLPerf benchmarks

---

### 八、数据资源

#### 上海交大知识图谱数据

http://acemap.sjtu.edu.cn/app/AceKG/

AceKG describes 114.30 million academic entities based on a consistent ontology, including 61,704,089 papers, 52,498,428 authors, 50,233 research fields, 19,843 academic institutes, 22,744 journals, 1,278 conferences and 3 special affiliations. In total, AceKG consists of 2.2 billion pieces of relationship information.

#### 伯克利开放驾驶视频数据集

http://bdd-data.berkeley.edu/

UC Berkeley 发布了迄今为止规模最大、最多样化的开放驾驶视频数据集——BDD100K。该数据集共包含 10 万个视频，BAIR 研究者在视频上采样关键帧，并为这些关键帧提供注释。此外，BAIR 还将在 CVPR 2018 自动驾驶 Workshop 上基于其数据举办三项挑战赛。

---
