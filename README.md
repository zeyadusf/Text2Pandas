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
 <a href="#">
  <img src="https://img.shields.io/badge/-HF Space -ffd21e?style=flat&logo=huggingface&logoColor=black" alt="Huggingface" />
</a>
</div>

<!--table of content-->
## Table of Content:
- ..
- ..
- ..

--- 

<!-- Problem definition -->
<br>

# Problem Definition:
The Text-to-Pandas task involves translating natural language text instructions into code that performs operations using the Pandas library, a powerful data manipulation and analysis tool in Python. The Pandas library allows users to handle data structures like DataFrames, perform filtering, aggregation.

In this task, given a natural language command, the goal is to generate the correct Pandas code to achieve the desired result on a dataset, such as filtering rows, computing aggregates, or selecting specific columns.

<!-- About Data --> 

# About Data :  
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
<!-- About Model -->
