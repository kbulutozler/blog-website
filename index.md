## Detecting Incivil Language in the Internet with Transformers (BERT)

The internet has lots of incivil language use since the age of forums. This tool basically uses a pre trained language model to detect incivil languages in the internet posts like tweets and comments. This tool has 2 different approaches, simple and multi domain. In the simple approach, the model is fine tuned to binary classify incivil language in the dataset. In multi domain approach, the datasets from multiple domains are used to make the model more robust for detecting incivil language bits from different domains. The code, the results and the paper will be provided.

### Task

Incivil language detection is basically a text classification task in NLP. The input is a piece of text, the output is a label. In the case of binary classification, the label is either 0 (no incivility) or 1 (yes incivility). In the case of multi label classification (not in this blog post, but in the shared paper), the output is a binary vector where ith element of the vector is 1 if the given input contains ith type of incivility (namecalling, vulgarity, threat etc.). The function to be aimed to learn in this task is h(repr(x)) = y. repr(x) is a tensor representing the input(a series of word vectors), y is the output vector. 

### Simple Approach

Language model based transfer learning is the basis for all the research has been done in this graduation project. Due to its huge impact in transfer learning in NLP, BERT has been used. There are multiple versions of how to transfer knowledge learnt in one task to another task. The version applied in this project can be summed up as:
- Prepare the dataset and task
- Get a widely used pre-trained model
- Adjust the model according to the task (fine-tune)
- Set the hyperparameters
- Train the model (weights actually)
- Test the model and report the results

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

