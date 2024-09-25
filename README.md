<div align="center">
  
# Text to Pandas
Convert Text with context about your dataframe to code Pandas by py  

<!-- related links (notebook - huggingface model - huggingface data - huggingface space)-->
 <a href="https://www.kaggle.com/code/zeyadusf/text-2-pandas-t5">
  <img src="https://img.shields.io/badge/-kaggle notebook-20bee1?style=flat&logo=kaggle&logoColor=black" alt="Kaggle" />
</a>
|
 <a href="https://www.kaggle.com/datasets/zeyadusf/convert-text-to-pandas">
  <img src="https://img.shields.io/badge/-kaggle Dataset-20bee1?style=flat&logo=kaggle&logoColor=black" alt="Kaggle" />
</a>
|
 <a href="https://huggingface.co/zeyadusf/text2pandas-T5">
  <img src="https://img.shields.io/badge/-HF model T5-ffd21e?style=flat&logo=huggingface&logoColor=black" alt="Huggingface" />
</a>
|
 <a href="https://huggingface.co/datasets/zeyadusf/text2pandas">
  <img src="https://img.shields.io/badge/-HF Dataset -ffd21e?style=flat&logo=huggingface&logoColor=black" alt="Huggingface" />
</a>
|
 <a href="https://huggingface.co/spaces/zeyadusf/Text2Pandas">
  <img src="https://img.shields.io/badge/-HF Space -ffd21e?style=flat&logo=huggingface&logoColor=black" alt="Huggingface" />
</a>
</div>

<!--table of content-->
## Table of Content:
1. [Problem Definition](#problem-definition)
2. [About Data](#about-data)
3. [About Model](#about-model)
4. [Inference Model](#inference-model)
5. [Future work](#future-work)
6. [Related Repositories](#related-repositories)


--- 

<!-- Problem definition -->
<br>

# Problem Definition
The Text-to-Pandas task involves translating natural language text instructions into code that performs operations using the Pandas library, a powerful data manipulation and analysis tool in Python. The Pandas library allows users to handle data structures like DataFrames, perform filtering, aggregation.

In this task, given a natural language command, the goal is to generate the correct Pandas code to achieve the desired result on a dataset, such as filtering rows, computing aggregates, or selecting specific columns.

<!-- About Data --> 

# About Data
I found two datasets about converting text with context to pandas code on Hugging Face, but the challenge is in the context. The context in both datasets is different which reduces the results of the model. First let's mention the data I found and then show examples, solution and some other problems.

- `Rahima411/text-to-pandas` :
    - The data is divided into Train with 57.5k and Test with 19.2k.

    - The data has two columns as you can see in the example:

       -  **"Input"**: Contains the context and the question together, in the context it shows the metadata about the data frame.
        - **"Pandas Query"**: Pandas code
```txt
              Input                                        |                     Pandas Query
-----------------------------------------------------------|-------------------------------------------
Table Name: head (age (object), head_id (object))          |   result = management['head.age'].unique()
Table Name: management (head_id (object),                  |
 temporary_acting (object))                                | 
What are the distinct ages of the heads who are acting?    |
  ```
- `hiltch/pandas-create-context`:
    - It contains 17k rows with three columns:
        - **question** : text .
        - **context** : Code to create a data frame with column names, unlike the first data set which contains the name of the data frame, column names and data type.
        -  **answer** : Pandas code.
      
```txt
           question                     |                        context                         |              answer 
----------------------------------------|--------------------------------------------------------|---------------------------------------
What was the lowest # of total votes?	| df = pd.DataFrame(columns=['_number_of_total_votes'])  |  df['_number_of_total_votes'].min()		
```

As you can see, **the problem** with this data is that they are **not similar as inputs and the structure of the context is different** . **My solution** to this problem was:
- Convert the first data set to become like the second in the context. I chose this because it is difficult to get the data type for the columns in the second data set. It was easy to convert the structure of the context from this shape `Table Name: head (age (object), head_id (object))` to this `head = pd.DataFrame(columns=['age','head_id'])` through this code that I wrote.
- Then separate the question from the context. This was easy because if you look at the data, you will find that the context always ends with "(" and then a blank and then the question.
You will find all of this in this code.
- You will also notice that more than one code or line can be returned to the context, and this has been engineered into the code.
```py
def extract_table_creation(text:str)->(str,str):
    """
    Extracts DataFrame creation statements and questions from the given text.
    
    Args:
        text (str): The input text containing table definitions and questions.
        
    Returns:
        tuple: A tuple containing a concatenated DataFrame creation string and a question.
    """
    # Define patterns
    table_pattern = r'Table Name: (\w+) \(([\w\s,()]+)\)'
    column_pattern = r'(\w+)\s*\((object|int64|float64)\)'
    
    # Find all table names and column definitions
    matches = re.findall(table_pattern, text)
    
    # Initialize a list to hold DataFrame creation statements
    df_creations = []
    
    for table_name, columns_str in matches:
        # Extract column names
        columns = re.findall(column_pattern, columns_str)
        column_names = [col[0] for col in columns]
        
        # Format DataFrame creation statement
        df_creation = f"{table_name} = pd.DataFrame(columns={column_names})"
        df_creations.append(df_creation)
    
    # Concatenate all DataFrame creation statements
    df_creation_concat = '\n'.join(df_creations)
    
    # Extract and clean the question
    question = text[text.rindex(')')+1:].strip()
    
    return df_creation_concat, question
```
After both datasets were similar in structure, they were merged into one set and divided into _72.8K_ train  and _18.6K_ test. We analyzed this dataset and you can see it all through the **[`notebook`](https://www.kaggle.com/code/zeyadusf/text-2-pandas-t5#Exploratory-Data-Analysis(EDA))**, but we found some problems in the dataset as well, such as
> - `Answer` : `df['Id'].count()` has been repeated, but this is possible, so we do not need to dispense with these rows.
> - `Context` : We see that it contains `147` rows that do not contain any text. We will see Through the experiment if this will affect the results negatively or positively.
> - `Question` : It is clear that the question has an invalid sample such as `?` etc.
> - `context_length` confirms that there are rows that contain `no text` and it is also clear that there are rows that contain `short text that is probably invalid`.
> - `question_length` is the same as context_length there are rows that contain `no text` and it is also clear that there are rows that contain `short text that is probably invalid`.

> **To address the problem, we need to filter out rows based on multiple criteria:** <br> *1. Remove rows with an invalid question, such as those containing ?.*<br>  *2. Remove rows where context or question lengths are less than 10, which indicates too short or invalid text.*<br>  *3. Retain the rows where the answer column may have repeated values, as this is expected behavior.*<br>  *4. Drop duplicate rows*

**You can find the data after structuring, filtering and cleaning through:** 
 * **[`Huggingface`](https://huggingface.co/datasets/zeyadusf/text2pandas)**
 * **[`Kaggle`](https://www.kaggle.com/datasets/zeyadusf/convert-text-to-pandas)**

- **number of characters**
![image](https://github.com/user-attachments/assets/e0859746-205d-453b-8ea6-ca7f0b2814af)
<br>

- **WordCloud**
![image](https://github.com/user-attachments/assets/2a9d13d7-ed1e-43dc-aea4-72c30ac4cbf2)

<!-- About Model -->

# About Model

I fine tuned **T5**, T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.
Using Transformers library and trained on _5 epochs_ and learning rate was _3e-5_ and scheduler type was _cosine_. You can see the rest of the hyperparameters in the [`notebook`](https://www.kaggle.com/code/zeyadusf/text-2-pandas-t5).<br>
**As for the results on [test dataset](https://huggingface.co/datasets/zeyadusf/text2pandas/viewer/default/test):**
>  1. **Prediction Loss: 0.0463**
   - _This is the average loss during the prediction phase of your model on the test set. A lower loss indicates that the model is predicting outputs that are closer to the expected values. In this case, a loss of 0.0463 suggests that the model is making fairly accurate predictions, as a low loss generally signals better performance._
> 2. **Prediction ROUGE-1: 0.8396**
   - _ROUGE-1 measures the overlap of unigrams (single words) between the predicted text and the reference text (in this case, the generated Pandas code and the ground truth). A score of 0.8396 (or ~84%) indicates that there is a high level of overlap between the predicted and true sequences, meaning that the model is capturing the general structure well._
> 3. **Prediction ROUGE-2: 0.8200**
   - _ROUGE-2 evaluates bigram (two-word) overlap between the predicted and reference texts. A score of 0.82 (~82%) suggests that the model is also doing well at capturing the relationships between words, which is important for generating coherent and syntactically correct code._
> 4. **Prediction ROUGE-L: 0.8396**
   - _ROUGE-L measures the longest common subsequence (LCS) between the predicted and reference sequences, focusing on the sequence order. A high ROUGE-L score (~84%) means the model is generating sequences that align well with the true code in terms of overall structure and ordering of operations. This is crucial when generating code, as the order of operations affects the logic._
> 5. **Prediction BLEU: 0.4729**
   - _BLEU evaluates how many n-grams (in this case, code snippets) in the predicted output match those in the reference output. A BLEU score of 0.4729 (or ~47%) is a moderate result for a text-to-code task. BLEU can be more challenging to optimize for code generation since it requires exact matches at a token level, including symbols, syntax, and even whitespace._

> **In general, this is a promising result, showing that the model is performing well on the task, with room for improvement on exact token matching (reflected by the BLEU score).** <br>

## Inference Model

```py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("zeyadusf/text2pandas-T5")
model = AutoModelForSeq2SeqLM.from_pretrained("zeyadusf/text2pandas-T5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_pandas(question, context, model, tokenizer, max_length=512, num_beams=4, early_stopping=True):
    """
    Generates text based on the provided question and context using a pre-trained model and tokenizer.

    Args:
        question (str): The question part of the input.
        context (str): The context (e.g., DataFrame description) related to the question.
        model (torch.nn.Module): The pre-trained language model (e.g., T5).
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        max_length (int): Maximum length of the generated text.
        num_beams (int): The number of beams for beam search.
        early_stopping (bool): Whether to stop the beam search when enough hypotheses have reached the end.

    Returns:
        str: The generated text decoded by the tokenizer.
    """
    # Prepare the input text by combining the question and context
    input_text = f"<question> {question} <context> {context}"

    # Tokenize the input text, convert to tensor, and truncate if needed
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_length)

    # Move inputs and model to the appropriate device
    inputs = inputs.to(device)
    model = model.to(device)

    # Generate predictions without calculating gradients
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)

    # Decode the generated tokens into text, skipping special tokens
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_text

# Example usage
question = "what is the total amount of players for the rockets in 1998 only?"
context = "df = pd.DataFrame(columns=['player', 'years_for_rockets'])"

# Generate and print the predicted text
predicted_text = generate_pandas(question, context, model, tokenizer)
print(predicted_text)
```
**output**
```py
df[df['years_for_rockets'] == '1998']['player'].count()

```

> The model link is at the top of the page.

# Future work
* Improve T5 results.
* Finetune another model such as _Llama2_

---

### Related Repositories

**[`LLMs from Scratch`](https://github.com/zeyadusf/LLMs-from-Scratch)**
**[`FineTuning Large Language Models`](https://github.com/zeyadusf/FineTuning-LLMs)**
**[`Topics in NLP and LLMs`](https://github.com/zeyadusf/topics-in-nlp-llm)**
<!--social media-->
<hr>

## 📞 Contact :

<div align="center">
<a href="https://www.kaggle.com/zeyadusf" target="blank">
  <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="zeyadusf" height="30" width="40" />
</a>


<a href="https://huggingface.co/zeyadusf" target="blank">
  <img align="center" src="https://github.com/zeyadusf/zeyadusf/assets/83798621/5c3db142-cda7-4c55-bcce-cc09d5b3aa50" alt="zeyadusf" height="40" width="40" />
</a> 

 <a href="https://github.com/zeyadusf" target="blank">
   <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="zeyadusf" height="30" width="40" />
 </a>
  
<a href="https://www.linkedin.com/in/zeyadusf/" target="blank">
  <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linkedin/linkedin-original.svg" alt="Zeyad Usf" height="30" width="40" />
  </a>
  
  
  <a href="https://www.facebook.com/ziayd.yosif" target="blank">
    <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg" alt="Zeyad Usf" height="30" width="40" />
  </a>
  
<a href="https://www.instagram.com/zeyadusf" target="blank">
  <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="zeyadusf" height="30" width="40" />
</a> 
</div>
