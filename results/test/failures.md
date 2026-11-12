## 1. What was previous single-system state of the art result on the CHiME-5 data?

Paper: 1909.12208. Evidence rank with para_200 + dense: 17 of 36

**Gold evidence 1:** To facilitate comparison with the recently published top-line in BIBREF12 (H/UPB), we have conducted a more focused set of experiments whose results are depicted in Table TABREF14. As explained in Section SECREF16, we opted for BIBREF12 instead of BIBREF13 as baseline because the former system is stronger. The experiments include refining the GSS enhancement using time annotations from ASR output (GSS w/ ASR), performing discriminative training on top of the AMs trained with LF-MMI and performing RNN LM rescoring. All the above helped further improve ASR performance. We report performance of our system on both single and multiple array tracks. To have a fair comparison, the results are compa

**Gold evidence 2:** Finally, cleaning up the training set not only boosted the recognition performance, but managed to do so using a fraction of the training data in BIBREF12, as shown in Table TABREF15. This translates to significantly faster and cheaper training of acoustic models, which is a major advantage in practice.

**Retrieved #1:** Using a single acoustic model trained with 308hrs of training data, which resulted after applying multi-array GSS data cleaning and a three-fold speed perturbation, we achieved a WER of 41.6% on the development (DEV) and 43.2% on the evaluation (EVAL) test set of CHiME-5, if the test data is also enhanced with multi-array GSS. This compares very favorably with the recently published top-line in BIBREF12, where the single-system best result, i.e., the WER without system combination, was 45.1% and 47.3% on DEV and EVAL, respectively, using an augmented training data set of 4500hrs total.

**Retrieved #2:** Depending on the number of available arrays for CHiME-5, two flavours of GSS enhancement were used in this work. In the single array track, all 4 channels of the array are used as input ($M = 4$), and the system is referred to as GSS1. In the multi array track, all six arrays are stacked to form a 24 channels super-array ($M = 24$), and this system is denoted as GSS6. The baseline time synchronization provided by the challenge organizers was sufficient to align the data for GSS6.

**Category:** Questionable label

---

## 2. Which other models that incorporate meta information do they compare against?

Paper: 1709.06365. Evidence rank with para_200 + dense: 26 of 67

**Gold evidence 1:** DMR, LDA with Dirichlet Multinomial Regression BIBREF8 : a model that can use multiple document labels. The Mallet implementation of DMR based on SparseLDA was used. Following Mallet, we set the mean of INLINEFORM0 to 0.0 and set the variances of INLINEFORM1 for the default label and the document labels to 100.0 and 1.0 respectively.

**Gold evidence 2:** LDA BIBREF0 : the baseline model. The Mallet implementation of SparseLDA BIBREF30 is used.

**Gold evidence 3:** LF-LDA, Latent Feature LDA BIBREF5 : a model that incorporates word embeddings. The original implementation was used. Following the paper, we used 1500 and 500 MCMC iterations for initialisation and sampling respectively and set INLINEFORM0 to 0.6, and used the original 50-dimensional GloVe word embeddings as word features.

**Gold evidence 4:** PTM, Pseudo document based Topic Model BIBREF18 : a model for short text analysis. The original implementation was used. Following the paper, we set the number of pseudo documents to 1000 and INLINEFORM0 to 0.1.

**Gold evidence 5:** WF-LDA, Word Feature LDA BIBREF16 : a model with word features. We implemented it on top of Mallet and used the default settings in Mallet for the optimisation.

**Gold evidence 6:** LLDA, Labelled LDA BIBREF11 and PLLDA, Partially Labelled LDA BIBREF9 : two models that make use of multiple document labels. The original implementation is used.

**Gold evidence 7:** GPU-DMM, Generalized Pólya Urn DMM BIBREF7 : a model that incorporates word semantic similarity. The original implementation was used. The word similarity was generated from the distances of the word embeddings. Following the paper, we set the hyper-parameters INLINEFORM0 and INLINEFORM1 to 0.1 and 0.7 respectively, and the symmetric document Dirichlet prior to INLINEFORM2 .

**Retrieved #1:** the meta information. Extensive experiments on several real world datasets demonstrate that our model achieves comparable or improved performance in terms of both perplexity and topic quality, particularly in handling sparse texts. In addition, compared with other models using meta information, our model runs significantly faster.

**Retrieved #2:** In this paper, we have presented a topic modelling framework named MetaLDA that can efficiently incorporate document and word meta information. This gains a significant improvement over others in terms of perplexity and topic quality. With two data augmentation techniques, MetaLDA enjoys full local conjugacy, allowing efficient Gibbs sampling, demonstrated by superiority in the per-iteration running time. Furthermore, without losing generality, MetaLDA can work with both regular texts and short texts. The improvement of MetaLDA over other models that also use meta information is more remarkable, particularly when the word-occurrence information is insufficient. As MetaLDA takes a particular 

**Category:** Summary-section pull

---

## 3. Are the recommendations specific to a region?

Paper: 2002.06854. Evidence rank with para_200 + dense: 29 of 30

**Gold evidence 1:** Relating to the items, there are $365\,056$ hotels, which is roughly 60 times smaller than the number of users. This ratio is also consistent with other datasets BIBREF14, BIBREF15.

**Retrieved #1:** In future work, we could easily increase the dataset with other languages and use it for multilingual recommendation. We release HotelRec for further research: https://github.com/Diego999/HotelRec.

**Retrieved #2:** We evaluated different models on the HotelRec subsets, 5-core and 20-core, on two tasks: rating prediction and recommendation performance. We have separated the evaluation because most models are only tailored for one of the tasks but not both. Therefore, we applied different models for each task and evaluated them separately.

**Category:** Questionable label

---

## 4. Is deep learning the state-of-the-art method in automated abuse detection

Paper: 1908.06024. Evidence rank with para_200 + dense: 47 of 65

**Gold evidence 1:** Mishra et al. mishra constructed a community graph of all users whose tweets are included in the data-twitter-wh dataset. Nodes in the graph were users while edges the follower-following relationship between them on Twitter. They then applied node2vec BIBREF21 to this graph to generate user embeddings. Inclusion of these embeddings into character n-gram based baselines yielded state of the art results on data-twitter-wh (F INLINEFORM0 increased from INLINEFORM1 and INLINEFORM2 to INLINEFORM3 and INLINEFORM4 on the racism and sexism classes respectively). The gains were attributed to the fact that user embeddings captured not only information about online communities, but also some elements o

**Retrieved #1:** In this paper, we survey the methods that have been developed for automated detection of online abuse, analyzing their strengths and weaknesses. We first describe the datasets that exist for abuse. Then we review the various detection methods that have been investigated by the NLP community. Finally, we conclude with the main trends that emerge, highlight the challenges that remain, outline possible solutions, and propose guidelines for ethics and explainability. To the best of our knowledge, this is the first comprehensive survey in this area. We differ from previous surveys BIBREF2 , BIBREF3 , BIBREF4 , BIBREF5 in the following respects: 1) we discuss the categorizations of abuse based on 

**Retrieved #2:** Abuse on the Internet represents an important societal problem of our time. Millions of Internet users face harassment, racism, personal attacks, and other types of abuse on online platforms. The psychological effects of such abuse on individuals can be profound and lasting. Consequently, over the past few years, there has been a substantial research effort towards automated abuse detection in the field of natural language processing (NLP). In this paper, we present a comprehensive survey of the methods that have been proposed to date, thus providing a platform for further development of this area. We describe the existing datasets and review the computational approaches to abuse detection, 

**Category:** Implicit answer

---

## 5. what regularisation methods did they look at?

Paper: 1707.05589. Evidence rank with para_200 + dense: 21 of 29

**Gold evidence 1:** Dropout is applied to feedforward connections denoted by dashed arrows in the figure. From the bottom up: to embedded inputs (input dropout), to connections between layers (intra-layer dropout), to the combined and the down-projected outputs (output dropout). All these dropouts have random masks drawn independently per time step, in contrast to the dropout on recurrent states where the same mask is used for all time steps in the sequence.

**Gold evidence 2:** The same dropout variants are applied to all three model types, with the exception of intra-layer dropout which does not apply to RHNs since only the recurrent state is passed between the layers. For the recurrent states, all architectures use either variational dropout BIBREF11 or recurrent dropout BIBREF12 , unless explicitly noted otherwise.

**Retrieved #1:** On two of the three datasets, we improved previous results substantially by careful model specification and hyperparameter optimisation, but the improvement for RHNs is much smaller compared to that for LSTMs. While it cannot be ruled out that our particular setup somehow favours LSTMs, we believe it is more likely that this effect arises due to the original RHN experimental condition having been tuned more extensively (this is nearly unavoidable during model development). Naturally, NAS benefitted only to a limited degree from our tuning, since the numbers of BIBREF1 were already produced by employing similar regularisation methods and a grid search. The small edge can be attributed to the 

**Retrieved #2:** Although we can draw attention to this problem, this paper does not offer a practical methodological solution beyond establishing reliable baselines that can be the benchmarks for subsequent work. Still, we demonstrate how, with a huge amount of computation, noise levels of various origins can be carefully estimated and models meaningfully compared. This apparent tradeoff between the amount of computation and the reliability of results seems to lie at the heart of the matter. Solutions to the methodological challenges must therefore make model evaluation cheaper by, for instance, reducing the number of hyperparameters and the sensitivity of models to them, employing better hyperparameter opt

**Category:** Wording mismatch

---

## 6. What is the accuracy of this model compared to sota?

Paper: 1904.09131. Evidence rank with para_200 + dense: 15 of 30

**Gold evidence 1:** We review the main differences between Wikidata and static knowledge bases extracted from Wikipedia, and analyze their implactions for entity linking. We illustrate these differences by building a simple entity linker, OpenTapioca, which only uses data from Wikidata, and show that it is competitive with other systems with access to larger data sources for some tasks. OpenTapioca can be trained easily from a Wikidata dump only, and can be efficiently kept up to date in real time as Wikidata evolves. We also propose tools to adapt existing entity linking datasets to Wikidata, and offer a new entity linking dataset, consisting of affiliation strings extracted from research articles.

**Retrieved #1:** We use the Wikidata JSON dump of 2018-02-24 for our experiments, indexed with Solr (Lucene). We restrict the index to humans, organizations and locations, by selecting only items whose type was a subclass of (P279) human (Q5), organization (Q43229) or geographical object (Q618123). Labels and aliases in all languages are added to a case-sensitive FST index. We trained our classifier and its hyper-parameters by five-fold cross-validation on the training sets of the ISTEX and RSS datasets. We used GERBIL BIBREF23 to evaluate OpenTapioca against other approaches. We report the InKB micro and macro F1 scores on test sets, with GERBIL's weak annotation match method.

**Retrieved #2:** The weighted graph $G_d$ can be represented as an adjacency matrix. We transform it into a column-stochastic matrix $M_d$ by normalizing its columns to sum to one. This defines a Markov chain on the candidate entities, that we will use to propagate the local evidence. BIBREF22 first combine the local features into a local evidence score, and then spread this local evidence using the Markov chain: $$ G(d) = (\alpha I + (1 - \alpha ) M_d)^k \cdot LC(d)$$ (Eq. 14)

**Category:** Answer in a table

---

## 7. Do the authors give examples of the core vocabulary which follows the scaling relationship of the bulk text?

Paper: 1903.04329. Evidence rank with para_200 + dense: 10 of 31

**Gold evidence 1:** We sorted the words falling into the "linear" scaling category according to their INLINEFORM0 values showing the goodness of fit for the fixed INLINEFORM1 model. The first 50 words in Table TABREF12 according to this ranking are some of the most common words of the English language, apart from some swearwords and abbreviations (e.g. lol) that are typical for Twitter language BIBREF10 . These are the words that are most homogeneously present in the text of all urban areas.

**Retrieved #1:** In this paper, we investigated the scaling relations in citywise Twitter corpora coming from the Metropolitan and Micropolitan Statstical Areas of the United States. We could observe a slightly superlinear scaling decreasing with the city population for the total volume of the tweets and words created in a city. When observing the scaling of individual words, we found that a certain core vocabulary follows the scaling relationship of that of the bulk text, but most words are sensitive to city size, and their frequencies either increase at a higher or a lower rate with city size than that of the total word volume. At both ends of the spectrum, the meaning of the most superlinearly or most sub

**Retrieved #2:** Scaling properties of language are a useful tool for understanding generative processes in texts. We investigate the scaling relations in citywise Twitter corpora coming from the Metropolitan and Micropolitan Statistical Areas of the United States. We observe a slightly superlinear urban scaling with the city population for the total volume of the tweets and words created in a city. We then find that a certain core vocabulary follows the scaling relationship of that of the bulk text, but most words are sensitive to city size, exhibiting a super- or a sublinear urban scaling. For both regimes we can offer a plausible explanation based on the meaning of the words. We also show that the paramet

**Category:** Summary-section pull

---

## 8. what is the monolingual baseline?

Paper: 1805.11598. Evidence rank with para_200 + dense: 14 of 21

**Gold evidence 1:** We use pretrained word embeddings as input to the model. For each of the shared task languages, we produced GloVe vectors BIBREF7 from the news, web, and Wikipedia text of the Leipzig Corpora Collection BIBREF8 . We trained 300-dimensional vectors, then reduced them to 100 dimensions with principal component analysis for efficiency.

**Gold evidence 2:** The input to the model consists of a sequence of pretrained embeddings for the surface forms of the sentence tokens. Each token embedding is also concatenated with a vector indicating whether the word is a predicate or not. Since the part-of-speech tags in the CoNLL 2009 dataset are based on a different tagset for each language, we do not use these. Each training instance consists of the annotations for a single predicate. These representations are then passed through a deep, multi-layer bidirectional LSTM BIBREF4 , BIBREF5 with highway connections BIBREF6 .

**Gold evidence 3:** We use the hidden representations produced by the deep biLSTM for both argument labeling and predicate sense disambiguation in a multitask setup; this is a modification to the models of He2017-deepsrl, who did not handle predicate senses, and of marcheggiani2017lstm, who used a separate model. These two predictions are made independently, with separate softmaxes over different last-layer parameters; we then combine the losses for each task when training. For predicate sense disambiguation, since the predicate has been identified, we choose from a small set of valid predicate senses as the tag for that token. This set of possible senses is selected based on the training data: we map from lemm

**Gold evidence 4:** Our basic model adapts the span-based dependency SRL model of He2017-deepsrl. This adaptation treats the dependent arguments as argument spans of length 1. Additionally, BIO consistency constraints are removed from the original model— each token is tagged simply with the argument label or an empty tag. A similar approach has also been proposed by marcheggiani2017lstm.

**Retrieved #1:** The standard approach to multilingual NLP is to design a single architecture, but tune and train a separate model for each language. While this method allows for customizing the model to the particulars of each language and the available data, it also presents a problem when little data is available: extensive language-specific annotation is required. The reality is that most languages have very little annotated data for most NLP tasks.

**Retrieved #2:** We present our results in Table TABREF11 . We observe that simple polyglot training improves over monolingual training, with the exception of Czech, where we observe no change in performance. The languages with the fewest training examples (German, Japanese, Catalan) show the most improvement, while large-dataset languages such as Czech or Chinese see little or no improvement (Figure FIGREF10 ). The language ID model performs inconsistently; it is better than the simple polyglot model in some cases, including Czech, but not in all. The language-specific LSTMs model performs best on a few languages, such as Catalan and Chinese, but worst on others. While these results may reflect differences 

**Category:** Wording mismatch

---

## 9. What is the size of this dataset?

Paper: 2002.06851. Evidence rank with para_200 + dense: 14 of 25

**Gold evidence 1:** In contrast, we propose a novel domain-specific dataset containing $14\,652$ samples, based on professional video game reviews obtained via Metacritic and gameplay sections from Wikipedia. By using Metacritic reviews in addition to Wikipedia articles, we benefit from a number of factors. First, the set of aspects used to assess a game is limited and consequently, reviews share redundancy. Second, because they are written by professional journalists, reviews tend to be in-depth and of high-quality. Additionally, when a video game is released, journalists have an incentive to write a complete review and publish it online as soon as possible to draw the attention of potential customers and incr

**Gold evidence 2:** We crawl approximately $265\,000$ professional reviews for around $72\,000$ games and $26\,000$ Wikipedia gameplay sections. Since there is no automatic mapping between a game to its Wikipedia page, we design some heuristics. The heuristics are the followings and applied in this order:

**Retrieved #1:** We use the standard ROUGE BIBREF4 used in summarization and report the ROUGE-L F1 score. ROUGE-L F1 is more appropriate to measure the quality of generated summaries in this context because summary lengths are longer than usual (see Table TABREF12) and vary across the dataset (see Table TABREF11). Another motivation to use ROUGE-L F1 is to compare abstractive models with extractive ones, as the output length is unknown a priori for the former, but not for the latter. We report in addition ROUGE-1 and ROUGE-2 recall scores.

**Retrieved #2:** To ensure consistent results across all comparative experiments, extractive models generate summaries of the same length as reference summaries. In realistic scenarios, summary lengths are not pre-defined and can be adjusted to produce different types of summaries (e.g., short, medium or long). We do not explicitly constrain the output length for abstractive models, as each summary is auto-regressively generated.

**Category:** Wording mismatch

---

## 10. Do they compare to previous work?

Paper: 1908.06893. Evidence rank with para_200 + dense: 44 of 52

**Gold evidence 1:** Comparison with a baseline - synthetic emails generated by Dada engine BIBREF6.

**Retrieved #1:** Sir I will really see if they were more comments tomorrow and review and act upon this evening $<$NET$>$. The engineer I can add there some $<$LINK$>$ there are the issues $<$NET$>$. Could you give me a basis for the call him he said The example above shows that while small substrings make some sense. The sequence of text fragments generated make very little sense when read as a whole. When comparing these with the phishing email structure described in BIBREF5, the generated emails have very little malicious content. The red text marks the incongruous text pieces that do not make sense. (B) Training on Legitimates + 5% Malicious content:

**Retrieved #2:** This kind of repetitive text generation was observed a number of times. However, we have not yet investigated the reasons for these repetitions. This could be an inherent problem of the LSTM model, or it could be because of the relatively small training dataset we have used. A third issue could be the temperature setting. More experiments are needed to determine the actual causes. The second aspect of error analysis is to look at the misclassification by the statistical detection algorithms. Here we look at a small sample of emails that were marked as legitimate despite being fake in nature. We try to investigate the factors in the example sample that can explain the misclassification errors

**Category:** Questionable label

---

## 11. What potential applications are demonstrated?

Paper: 1912.00342. Evidence rank with para_200 + dense: 41 of 58

**Gold evidence 1:** Since the proposed approach regards the formal semantics and the task domain is not specified, we expect our study to be meaningful for a general AI that talks with human beings without making the users feel isolated. Recalling that for also humans, the reaction towards the directive and the non-directive utterance differs, our two-way approach makes sense. Along with the non-task-oriented dialog, our scheme may be useful for avoiding inadvertent ignorance of the users' will.

**Gold evidence 2:** Beyond the application to the spoken language understanding (SLU) modules within the smart agents, our approach can be utilized in making up the paraphrase corpus or supporting the semantic web search. Along with the boosted performance of recent text generation and reconstruction algorithms, we expect a large size of the dataset is furthermore constructed and be utilized with the real-life personal agents.

**Retrieved #1:** This research was supported by Projects for Research and Development of Police science and Technology under Center for Research and Development of Police science and Technology and Korean National Police Agency funded by the Ministry of Science, ICT and Future Planning (PA-J000001-2017-101). Also, this work was supported by the Technology Innovation Program (10076583, Development of free-running speech recognition technologies for embedded robot system) funded By the Ministry of Trade, Industry & Energy (MOTIE, Korea).

**Retrieved #2:** the police' (Figure 1). Although many advanced dialog managing systems may generate a plausible reaction to the input utterances, it is different from extracting the exact intent argument (a question set or a to-do-list) that should be investigated for an actual operation.

**Category:** Boilerplate Pull

---

## 12. Do the QA tuples fall under a specific domain?

Paper: 1910.13793. Evidence rank with para_200 + dense: 16 of 26

**Gold evidence 1:** For our models, we'll use a customer support dataset with a relatively high usage of emoji. The dataset contains 2000 tuples collected by BIBREF24 that are sourced from Twitter. They provide conversations, which consist of at least one question and one free-form answer. Some conversations are longer, in this case we ignored the previous context and only looked at the last tuple. This dataset illustrates that even when contacting companies, Twitter users keep using emoji relatively often, 8.75% of all utterances.

**Retrieved #1:** The proposed models in this field are diverse and largely depending on how the problem is formulated. When considering free-form responses, generative models like GPT BIBREF30, GPT-2 BIBREF6 or seq2seq BIBREF31 are appropriate. When the conversational task is modeled as a response selection task to pick the correct response out of $N$ candidates BIBREF32, BIBREF26, BIBREF33, this can be a language model like BERT BIBREF5 with a dedicated head.

**Retrieved #2:** For the sentence prediction task, Table TABREF11 shows the results of the baseline and our model with additional emoji tokens. For each of the 600 utterance pairs of the held-out test set, we added 99 randomly selected negative candidates, as described in Subsection SECREF10. The 1-out-of-100 accuracy measures how often the true candidate was correctly selected and the mean rank gives an indication of how the model performs if it fails to correctly select the positive candidate. The baseline correctly picks 12.7% of all candidate responses, out of 100. Given that the dataset is focussed on support questions and multiple responses are likely to be relevant, this baseline already performs admi

**Category:** Questionable label

---

## 13. how were the data instances chosen?

Paper: 1907.08540. Evidence rank with para_200 + dense: 29 of 46

**Gold evidence 1:** Therefore, in order to target only those tweets that are rich in human activity content, we formulate a set of queries that allows us to use the Twitter Search API to find instances of users tweeting about common human activities. Each query contains a first-person, past-tense verb within a phrase that describes a common activity that people do. Using this approach, we are able to retrieve a set of tweets that contains a high concentration of human activity content, and we also find that users who wrote these tweets are much more likely to have written other tweets that describe human activities (Table TABREF1 ). We build our set of human activity queries from two sources: the Event2Mind dat

**Retrieved #1:** We split our data at the user-level, and from our set of valid users we use 200,000 instances for training data, 10,000 as test data, and the rest as our validation set.

**Retrieved #2:** We consider two variations on our dataset: the first is a simplified, 50-class classification problem. We choose the 50 most common clusters out of our full set of INLINEFORM0 and only make predictions about users who have reportedly performed an activity in one of these clusters. The second variation uses the entire dataset, but rather than making predictions about all INLINEFORM1 classes, we only make fine-grained predictions about those classes for which INLINEFORM2 . We do this under the assumption that training an adequate classifier for a given class requires at least INLINEFORM3 examples. All classes for which INLINEFORM4 are assigned an “other” label. In this way, we still make a pre

**Category:** Wording mismatch

---

## 14. What are the baselines this paper uses?

Paper: 1810.02229. Evidence rank with para_200 + dense: 20 of 27

**Gold evidence 1:** Results for the experiments are illustrated in Table 2 . We also report the results of the best system that participated at EVENTI Subtask B, FBK-HLT BIBREF23 . FBK-HLT is a cascade of two SVM classifiers (one for detection and one for classification) based on rich linguistic features. Figure 1 plots charts comparing F1 scores of the network initialized with each of the five embeddings against the FBK-HLT system for the event detection and classification tasks, respectively.

**Retrieved #1:** The EVENTI corpus consists of three datasets: the Main Task training data, the Main task test data, and the Pilot task test data. The Main Task data are on contemporary news articles, while the Pilot Task on historical news articles. For our experiments, we focused only on the Main Task. In addition to the training and test data, we have created also a Main Task development set by excluding from the training data all the articles that composed the test data of the Italian dataset at the SemEval 2010 TempEval-2 campaign BIBREF6 . The new partition of the corpus results in the following distribution of the $<$ EVENT $>$ tag: i) 17,528 events in the training data, of which 1,207 are multi-token

**Retrieved #2:** Evaluation is conducted using the EVENTI evaluation framework. Standard Precision, Recall, and F1 apply for the event detection. Given that the extent of an event tag may be composed by more than one tokens, systems are evaluated both for strict match, i.e. one point only if all tokens which compose an $<$ EVENT $>$ tag are correctly identified, and relaxed match, i.e. one point for any correct overlap between the system output and the reference gold data. The classification aspect is evaluated using the F1-attribute score BIBREF7 , that captures how well a system identify both the entity (extent) and attribute (i.e. class) together.

**Category:** Wording mismatch

---

## 15. How is the gold standard defined?

Paper: 1605.05166. Evidence rank with para_200 + dense: 19 of 43

**Gold evidence 1:** For the purposes of this paper, we focused on matching accounts between two of the largest social networks: Twitter and Facebook. In order to proceed with our study, we needed a sizeable (few thousand) number of English speaking users with accounts on both Twitter and Facebook. We also needed to know the precise matching between the Twitter and Facebook accounts for our ground truth.

**Gold evidence 2:** To that end, we crawled publicly available, English-language, Google Plus accounts using the Google Plus API and scraped links to the users' other social media profiles. (Note that one of the reasons why we used Twitter and Facebook is that they were two of the most common sites linked to on Google Plus). We used a third party social media site (i.e., Google Plus), one that was not used in our analysis to compile our ground truth in order to limit selection bias in our data collection.

**Gold evidence 3:** We discarded all users who did not link to an account for both Twitter and Facebook and those whose accounts on either of these sites were not public. We then used the APIs of Twitter and Facebook to collect posts made by the users on these sites. We only collected the linguistic content and the date and time at the which the posts were made. For technical and privacy reasons, we did not collect any information from the profile of the users, such as the location, screen name, or birthday.

**Retrieved #1:** TF considers all terms as equally important, however, certain terms that occur too frequently should have little effect (for example, the term “the"). And conversely, terms that occur less in a document can be more relevant. Therefore, in order to weigh down the effects of the terms that occur too frequently and weigh up the effects of less frequently occurring terms, an Inverse Document Frequency factor is incorporated which diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely. Generally speaking, the Inverse Document Frequency is a measure of how much information a word provides, that is, whether the term is commo

**Retrieved #2:** TF-IDF can be thought of as a heuristic measure of the extent to which different words are characteristic of a user. We came up with a new, theoretically motivated measure of “being characteristic" for words. We considered the following setup : The whole corpus of the $11,224$ Twitter and Facebook accounts was treated as one long string; For each token in the string, we know the user who produced it. Imagine that we removed this information and are now making a guess as to who the user was. This will give us a probability distribution over all users;

**Category:** Wording mismatch

---

## 16. How do they interpret the model?

Paper: 2003.04707. Evidence rank with para_200 + dense: 11 of 60

**Gold evidence 1:** Applications of Neuro-symbolism ::: Application II: Neural Question-Answering using Commonsense Knowledge Bases ::: Knowledge elicitation

**Gold evidence 2:** ConceptNet. We identify ConceptNet relations that connect questions to the answer-options. The intuition is that these relation paths would provide explicit evidence that would help the model find the answer. Formally, given a question $Q$ and an answer-option $O$, we find all ConceptNet relations (C1, r, C2), such that $C1 \in Q$ and $C2 \in O$, or vice versa. This rule works well for single-word concepts. However, a large number of concepts in ConceptNet are actually phrases, where finding exactly matching phrases in $Q/O$ is more challenging. To fully utilize phrase-based relations, we relaxed the exact-match constraint to the following:

**Retrieved #1:** BREF32. The main downsides of this approach are that models are difficult to interpret and that they lack mechanisms for incorporating explicit commonsense knowledge. Conversely, purely knowledge-oriented approaches combine structured knowledge bases and perform symbolic reasoning, on the basis of axiomatic principles. Such models enjoy the property of interpretability, but often lack the ability to estimate the statistical salience of an inference, based on real-world observations. Hybrid models are those that attempt to fuse these two approaches, by extracting knowledge from structured knowledge bases and using the resulting information to guide the learning paradigm of statistical estimat

**Retrieved #2:** model is more accurate when it includes specialized modules to evaluate questions and candidate answers on the basis of a common knowledge graph. In both cases, explainability emerges as a property of the mechanisms that we implemented, through this combination of data-driven algorithms with the relevant knowledge resources.

**Category:** Summary-section pull

---

## 17. Which translation model do they employ?

Paper: 1610.06510. Evidence rank with para_200 + dense: 20 of 40

**Gold evidence 1:** We train subword level phrase-based SMT models between related languages. Along with BPE level, we also train PBSMT models at morpheme and OS levels for comparison.

**Retrieved #1:** Since we are concerned with low resource scenarios, a desirable property of subword units is robustness of the translation models to change of translation domain. kunchukuttan2016orthographic have shown that OS level models are robust to domain change. Since BPE units are learnt from a specific corpus, it is not guaranteed that they would also be robust to domain changes. To study the behaviour of BPE unit trained models, we also tested the translation models trained on tourism & health domains on an agriculture domain test set of 1000 sentences (see Table TABREF27 for results). In this cross-domain translation scenario, the BPE level model outperforms the OS-level and word-level models for 

**Retrieved #2:** While building phrase-based SMT models at the subword level, we use (a) monotonic decoding since related languages have similar word order, (b) higher order languages models (10-gram) since data sparsity is a lesser concern owing to small vocabulary size BIBREF11 , and (c) word level tuning (by post-processing the decoder output during tuning) to optimize the correct translation metric BIBREF8 . Following decoding, we used a simple method to regenerate words from subwords (desegmentation): concatenate subwords between consecutive occurrences of boundary marker characters. We trained translation systems over the following basic units: character, morpheme, word, orthographic syllable and BPE u

**Category:** Questionable label

---

## 18. Does the SESAME dataset include discontiguous entities?

Paper: 1908.05758. Evidence rank with para_200 + dense: 11 of 28

**Gold evidence 1:** The next step consists of detecting mentions to entities in the raw text. To do this, we tag character segments that exactly match one of the known names of an entity. For instance, we can tag two different entities in the following text:

**Gold evidence 2:** First, we tokenize only the words corresponding to mentions of an entity. In order to explicitly mark the boundaries of each entity, we use the BIO format, where we add the suffix “B” (begin) to the first token of a mention and “I” (inside) to the tokens following it. This gives us:

**Gold evidence 3:** Second, we tokenize the remaining text, as illustrated by the following example: $w_{i}$ denotes a word token, while $s_{i}$ corresponds to a sentence token.

**Gold evidence 4:** $ \underbrace{\text{John}}_{\text{B-PER}} \underbrace{\text{Smith}}_{\text{I-PER}} \text{travelled to} \underbrace{\text{Rio}}_{\text{B-LOC}} \underbrace{\text{de}}_{\text{I-LOC}} \underbrace{\text{Janeiro}}_{\text{I-LOC}} \text{. Visited } \underbrace{\text{Copacabana}}_{\text{B-LOC}} $

**Retrieved #1:** We only consider sentences that have annotated entities. After all, sentences with only parser extraction entities do not take advantage of the human discernment invested in the structuring of the data of DBpedia. SESAME consists of 3,650,909 sentences, with lengths (in terms of number of tokens) following the distribution shown in Figure FIGREF42. A breakdown of relevant statistics, such as the mean and standard deviation of sentences' lengths, is given in Table TABREF43. SESAME consists of 87,769,158 tokens in total. The count and proportion of each entity tag (not a named entity, organization, person, location) is given in TABREF45. Not surprisingly, the vast majority of words are not rel

**Retrieved #2:** Since our approach consists of matching raw text to a list of entity names, it does not account for context in which the entity was mentioned. For example, while under a specific context a country entity can exert the role of an organization, our method will tag it as a location regardless of the context. Therefore, our approach delimits an entity mention as a semantic object that does not vary in according to the context of the sentence. By following the above methodology on the Portuguese Wikipedia and DBpedia, we create a massive silver standard dataset for NER. We call this dataset SESAME (Silver-Standard Named Entity Recognition dataset). We then proceed to study relevant statistics of 

**Category:** Implicit answer

---

## 19. What domain of text are they working with?

Paper: 1603.07252. Evidence rank with para_200 + dense: 8 of 42

**Gold evidence 1:** One stumbling block to applying neural network models to extractive summarization is the lack of training data, i.e., documents with sentences (and words) labeled as summary-worthy. Inspired by previous work on summarization BIBREF7 , BIBREF13 and reading comprehension BIBREF9 we retrieve hundreds of thousands of news articles and corresponding highlights from the DailyMail website. Highlights usually appear as bullet points giving a brief overview of the information contained in the article (see Figure 1 for an example). Using a number of transformation and scoring algorithms, we are able to match highlights to document content and construct two large scale training datasets, one for senten

**Gold evidence 2:** We evaluate our models both automatically (in terms of Rouge) and by humans on two datasets: the benchmark DUC 2002 document summarization corpus and our own DailyMail news highlights corpus. Experimental results show that our summarizers achieve performance comparable to state-of-the-art systems employing hand-engineered features and sophisticated linguistic constraints.

**Retrieved #1:** Directions for future work are many and varied. One way to improve the word-based model would be to take structural information into account during generation, e.g., by combining it with a tree-based algorithm BIBREF31 . It would also be interesting to apply the neural models presented here in a phrase-based setting similar to lebret2015phrase. A third direction would be to adopt an information theoretic perspective and devise a purely unsupervised approach that selects summary sentences and words so as to minimize information loss, a task possibly achievable with the dataset created in this work. We would like to thank three anonymous reviewers and members of the ILCC at the School of Infor

**Retrieved #2:** In a nutshell, we retrieved hundreds of thousands of news articles and their corresponding highlights from DailyMail (see Figure 1 for an example). The highlights (created by news editors) are genuinely abstractive summaries and therefore not readily suited to supervised training. To create the training data for sentence extraction, we reverse approximated the gold standard label of each document sentence given the summary based on their semantic correspondence BIBREF7 . Specifically, we designed a rule-based system that determines whether a document sentence matches a highlight and should be labeled with 1 (must be in the summary), and 0 otherwise. The rules take into account the position o

**Category:** Questionable label

---

## 20. Do they have an elementary unit of text?

Paper: 1610.00479. Evidence rank with para_200 + dense: 10 of 80

**Gold evidence 1:** We define a nonsymbolic approach as one that is tokenization-free, i.e., no assumption is made that there are segmentation boundaries and that each segment (e.g., a word) should be represented (e.g., by a word embedding) in a way that is independent of the representations (e.g., word embeddings) of neighboring segments. Methods for training text representation models that require tokenized text include word embedding models like word2vec BIBREF1 and most group (ii) methods, i.e., character-level models like fastText skipgram BIBREF2 .

**Retrieved #1:** Of course, it is an open question whether nonsymbolic methods can perform better than morphological analysis, but the foregoing discussion motivates us to investigate them. chung16characternmt focus on problems with the tokens produced by segmentation algorithms. Equally important is the problem that tokenization fails to capture structure across multiple tokens. The job of dealing with cross-token structure is often given to downstream components of the pipeline, e.g., components that recognize multiwords and named entitites in English or in fact any word in a language like Chinese that uses no overt delimiters. However, there is no linguistic or computational reason in principle why we sho

**Retrieved #2:** Although such models learn/use character n-gram embeddings from tokenized text or short text segments, to represent a piece of text, the occurring character n-grams are usually summed without the need for tokenization. For example, the phrase “Berlin is located in Germany” is represented with character 4-grams as follows: “Berl erli rlin lin_ in_i n_is _is_ is_l s_lo _loc loca ocat cate ated ted_ ed_i d_in _in_ in_G n_Ge _Ger Germ erma rman many any.” Note that the input has not been tokenized and there are n-grams spanning token boundaries. We also include non-embedding approaches using bag-of-n-grams within this group as they go beyond word and token representations.

**Category:** Implicit answer

---

