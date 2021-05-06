## Detecting Incivil Language in the Internet with Transformers (BERT)

The internet has lots of incivil language use since the age of forums. This tool basically uses a pre trained language model to detect incivil languages in the internet posts like tweets and comments. This tool has 2 different approaches, simple and multi domain. In the simple approach, the model is fine tuned to binary classify incivil language in the dataset. In multi domain approach, the datasets from multiple domains are used to make the model more robust for detecting incivil language bits from different domains. The original paper can be found [here](https://www.aclweb.org/anthology/2020.alw-1.4/)

### Task

Incivil language detection is basically a text classification task in NLP. The input is a piece of text, the output is a label. In the case of binary classification, the label is either 0 (no incivility) or 1 (yes incivility). In the case of multi label classification (not in this blog post, but in the shared paper), the output is a binary vector where ith element of the vector is 1 if the given input contains ith type of incivility (namecalling, vulgarity, threat etc.). The function to be aimed to learn in this task is h(repr(x)) = y. repr(x) is a tensor representing the input(a series of word vectors), y is the output vector. 

### Simple Approach

Language model based transfer learning is the basis for all the work in this blog post. Due to its huge impact in transfer learning in NLP, BERT has been used. Simple approach is basically fine tuning BERT. BERT is taken and a fully connected layer on top of it is added. That fully connected layer has a softmax layer that will give the binary classification prediction. The image below is a summary. We can think of the outcome positive as incivility exists, negative as incivility does not exist. 

<img src="“images/image.jpg”" alt="hi" class="inline"/>

### Multi Domain Approach

This approach is based on simple approach, but using multiple datasets from different domains. Multidomain approach is also based on transfer learning with pre-trained language model. The goal is to research how dataset characteristics and annotation differences affects the robustness of the predictions when the datasets are used together. Domain can affect the content and the quality of the data. Some hypothetical domain examples to make the word “domain” clearer:

- 3 datasets about same topic (i.e. racism): one of them consists of tweets from Twitter, one of them consists of posts from Facebook, one of them consists of comments from a white supremacist forum. Each of the datasets represent a domain because even if the topic is same, the samples will be different based on:
  - each platform has its own slang/lingo
  - user groups have different backgrounds

- 2 datasets from same platform (i.e. twitter): one of them is about namecalling, other one is about misogyny. Each of them represent a domain based on:
  - namecalling can contain common misogyny terms but not necessarily
  - misogyny can contain common namecalling terms but not necessarily

The idea is that if we use datasets together, the classifier will benefit from variety of samples and robustness will increase. For example, twitter dataset has several samples that is actually racist, but the slang used in these samples is not commonly seen in the majority of the twitter dataset. Let’s say that particular slang exists in facebook and/or forum datasets. When the classifier is trained in all of the datasets, meaning the classifier is exposed to that particular slang more than the twitter data has, and when the classifier is tested on twitter dataset, it’s expected that its chances to detect that particular racist samples will increase.

Three methods are considered for training classifiers for prediction in multiple domains:
- Single
  - 1 classifier is fine-tuned for each domain
  - This is basically the simple approach above
  - This will be comparison baseline
- Joint
  - 1 classifier is fine-tuned on the combined training data from all domains
- Joint + Single
  - A joint classifier is fine-tuned
  - 1 single classifier for each domain is initialized by using parameters of the joint classifier
  - Those single classifiers are fine-tuned

### Data

#### Local news comments
A collection of comments taken from local news website. Following labels are used as classes:
- aspersion
- lying accusation
- name-calling
- pejorative
- vulgarity

#### Local Politics Tweets
A collection of microblog posts from the Twitter accounts of the local politicians. Only one label is used as class:
- name-calling

#### Russian troll Tweets
A small subset of the 3 million English Tweets written by Russian trolls. Only one label is used as class:
- name-calling

Since only 1 label (name-calling) is common in all 3 datasets, first dataset is adjusted to be used only for name-calling label. Some name-calling examples:

```markdown
"You pathetic, pimple-faced little adolescent moron. Have you not studied sarcasm yet? \n\nLet me guess...you are a product of home schooling, right?"

"Canada doesn’t need another CUCK! We already have enough #LooneyLeft #Liberals f**king up our great country! #Qproofs #TrudeauMustGo"

"if yo girl doesnt swallow kids, that bitch basic!"
```

And some civil examples which does not have name-calling:

```markdown
"im genuinely getting mad at this children's video game about mickey mouse and darkness and big keys"

"please advice about Bank nifty & how the NPA will benefit for PSU Banks ?"

"Strawberry Swing As an editor with a history of personal attacks and of creating attack pages, I disagree that it wasn't appropriate."
```
Data can be provided upon request from Steven Bethard from University of Arizona Information Department.

### Application

#### Source Code
The source code is used from for a version (0.6.1) of transformers repository. My modified version of the original code can be found from my own [repository](https://github.com/kbulutozler/BERT-multilabel-classification). transformers repository from huggingface is publicly available with all its versions on github.

#### Training
Once you clone the repository, only run_classifier_custom.py file is enough. The command should specify following parameters:
- data_dir
- bert_model: for fine tuning, use bert-base, for using already fine tuned model (multi domain approach), give the directory of said model
- task_name: give sst-2
- output_dir
- cache_dir
- max_seq_length: try 128 or 256
- train_batch_size: try 16, 32, 64 or 128
- learning_rate: try 8e-6, 2e-5, 4e-5 or 8e-5
- num_train_epochs: try 2, 3, 4, 5, 6 or 8

This is basically fine tuning a language model by adjusting it to a binary classification task (sst-2) for a specific domain (see data section). As explained in the approach sections, 

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

