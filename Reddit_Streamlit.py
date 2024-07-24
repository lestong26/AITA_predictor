
# Check if Streamlit is installed correctly, print version
import streamlit as st
print(st.__version__)

# 1: Seeing your data in Streamlit with st.write
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.few_shot import DynamicFewShotGPTClassifier
from openai import OpenAI
from skllm.models.gpt.text2text.summarization import GPTSummarizer


#Other Libraries
from bs4 import BeautifulSoup
from unidecode import unidecode
import re
import warnings
import joblib
import pickle

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(layout='wide')

api_key = st.secrets["api_key"]
SKLLMConfig.set_openai_key(api_key)
client = OpenAI(api_key=api_key)

# from skllm.preprocessing import GPTVectorizer
# vectorizer = GPTVectorizer()
# vectorizer.openai_key = 'api_key'


#predicition pipeline function
def predicition_pipeline(query):

  ### equate the pkl file here
  with open('dyfw_clfv2.pkl', 'rb') as f:
    model = pickle.load(f)
### Extract model prediction
  prediction = model.predict(pd.Series(query))
  model_prediction = prediction[0]      # predict if YTA or NTA

### Extract nearest neighbor comment and tag
  prompt = model._get_prompt(query)

# comment content
  start_index = prompt['messages'].find('Sample input')
  ending_index = prompt['messages'].find('Sample target', start_index+1)
  comment_content = prompt['messages'][start_index:ending_index]
  # comment_content

# gpt tagging: Sample target: yta\n\n\n
  start_index = prompt['messages'].find('Sample target')
  ending_index = prompt['messages'].find('Sample input', start_index+1)
  target_content = prompt['messages'][start_index:ending_index]
  # target_content

  def preprocess_text(text):
      if not isinstance(text, str):
          return text

      text = unidecode(text)

      pattern = re.compile(r'[\t\n]|<.*?>|!function.*;|\.igframe.*}')
      text = pattern.sub(' ', text)
      text = (
          text
          .replace('&#8217;', "'")
          .replace('&#8220;', '"')
          .replace('&#8221;', '"')
      )

      soup = BeautifulSoup(text, 'html.parser')
      cleaned_text = soup.get_text()
      return cleaned_text

  comment_cleaned = preprocess_text(comment_content)
  target_cleaned = preprocess_text(target_content)


  def generate_response(user_submission, prompt):
      response = client.chat.completions.create(
          model='gpt-3.5-turbo',
          messages=[
              {'role': 'system',
              'content':
              f"Perform the specified tasks based on this Am I the Asshole (AITAH) submission:\n\n{user_submission}"},
              {'role': 'user', 'content': prompt}
          ]
      )
      return response.choices[0].message.content

  def aitah_explanation(user_submission, predicted_tagging, nearest_comment, nearest_comment_tag):
    nearest_comment = nearest_comment.replace('Sample input:','')
    prompt = f"Briefly explain why you gave the {predicted_tagging} to the above user submission.\n\nShortly reference this similar submission with tagging {nearest_comment_tag} as a basis for your explanation: {nearest_comment}"
    explanation = generate_response(user_submission, prompt)
    return [prompt, explanation, nearest_comment, nearest_comment_tag]

  prompt_passed, gpt_feedback, similar_submission, similar_submission_tag = aitah_explanation(user_submission=query, predicted_tagging=prediction[0], nearest_comment=comment_cleaned, nearest_comment_tag=prediction[0])

  return [prompt_passed, gpt_feedback, similar_submission, similar_submission_tag]
    
###---------------pages code-------------------------

# Add page options in a sidebar
my_page = st.sidebar.radio('Page Navigation',
                           ['About our App', 'Exploratory Data Analysis', 
                            'How does the Model work?', 
                            'Check if You\'re the A**hole'
                           ])


if my_page == 'About our App':
    st.title("Bakit parang galit ka? Bakit kasalanan ko? Parang kasalanan ko.")
    st.subheader(body="Leveraging Data from the \"Am I the A**hole?\" Subreddit")
    st.markdown("This Streamlit app makes use of webscraped subreddit data to develop a machine learning model that can assist you in creating moral judgments in various situations you face in everyday life.")
    st.divider()

    st.header("Our Project")
    st.markdown('Understanding who we are and how others see us is key not only for job performance, career success, and leadership effectiveness but also for personal relationships. However, self-awareness is surprisingly rare in today’\s workplace. Harvard Business Review\'s five-year research study found that while 95% of people believe they are self-aware, only 10 to 15% truly are.')
    st.image('Reddit_Photos/Intro_Context.png', width = 700)
    st.markdown("And since many people struggle with self-awareness, this may lead to several challenges:\n\nStrained personal relationships: Without self-awareness, individuals may not realize how their actions impact their loved ones, causing misunderstandings, hurt feelings, and damaged relationships.\n\n**Lack of empathy**: Without self-awareness, it's difficult to put oneself in another person's shoes and understand their perspective, which can lead to insensitive or hurtful behavior in personal relationships.\n\n**Inability to learn and grow**: A lack of self-awareness can prevent people from recognizing their weaknesses and areas for improvement, hindering personal growth and development.\n\n**Difficulty in conflict resolution**: Without understanding oneself, it becomes challenging to navigate and resolve conflicts effectively.")
    

    st.divider()
    st.header("What's the goal?")
    st.markdown('''Our project focuses on building a machine learning model using data scraped from two am-i-the-a\*\*hole subreddits. These subreddits are full of real-life stories where people seek judgment on their actions, helping us determine if someone's behavior might be seen as "a\*\*hole" (yta) or not (nta). The aim is to create a tool that promotes self-awareness among users by offering insights into their decisions. Studies show that many people underestimate their own self-awareness, and our model seeks to bridge that gap by encouraging reflection and providing a moral compass for personal behavior.''')
    st.image('Reddit_Photos/Intro_Objective.png')
    


    st.header("Our Data")
    st.markdown("Our community has around 1.9 million members who share and discuss their interpersonal conflicts. According to the site, it's 'a catharsis for the frustrated moral philosopher in all of us, and a place to finally find out if you were wrong in an argument that's been bothering you.'\n\nWe focus on two main labels: YTA (You're the Asshole), which means you were in the wrong, acting unethically or inconsiderately, and NTA (Not the Asshole), indicating you were justified and appropriate in your actions.")
    
    col1, col2 = st.columns(2)

    # Display images in each column
    with col1:
        st.image('Reddit_Photos/Intro_Data_2.png', use_column_width=True)
    
    with col2:
        st.image('Reddit_Photos/Intro_Data.png', use_column_width=True)
    st.markdown('''Now, let's dive into the data we've gathered from these subreddit posts. We analyzed 5.9k posts from 2018 to 2024. Our dataset includes columns like 'selftext' (the content of the post), filtered comments, YTA count, NTA count, and the label assigned to the post or the original poster. Additionally, we gathered information like the author, creation dates, upvote ratio, and URL, but we’ve omitted these from our main analysis.\n\nStay tuned as we explore the insights from this data!''')
    
    # st.markdown("Note for label column: nta = \'not the a\*\*hole\' | yta = \'you\'re the a\*\*hole\'")
    df = pd.read_csv("fewshot_df.csv")
    st.write(df.head())
    st.divider()
    
    st.header("Just some Limitations")
    st.markdown('''For our analysis, we used a few key tools:\nPython for data processing.\nPRAW (Python Reddit API Wrapper) to fetch the posts for our dataset.\n OpenAI for scoping.''')
    st.markdown('''However, there are some limitations to our study. We only considered the YTA (You’re the Asshole) and NTA (Not the Asshole) labels, excluding the NAH (No Assholes Here) and ESH (Everyone Sucks Here) labels. Additionally, we evaluated our model using accuracy, BLEU, METEOR, and ROUGE metrics. It's also important to note that the OpenAI API key tokens and the Reddit API have their own limitations.''')
    st.image('Reddit_Photos/Intro_Limitation.png', width = 700)
    st.markdown("We hope this gives you a clear picture of our approach and the scope of our study!")
    


elif my_page == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')
    st.markdown('Allow us to take you through the findings we obtained from our webscraped data')
    st.divider()

    st.header('Upvotes, Upload Schedules and Comments')

    st.subheader('Average Upvote Ratio for the Last 5 Years')
    st.image('Reddit_Photos/EDA_trend.png', caption = 'The highest average upvote ratio of 0.97 was recorded in September 2021', width = 800)
    st.subheader('Daily and Hourly Trends of Submissions')
    st.image('Reddit_Photos/EDA_Submissions.png', caption = 'There is an increase in the volume of submissions between noon and 7:00 in the evening, with peak activity during the start of the week ', width = 700)
    st.subheader('Relationship Between Upvote Ratio & Comment Count')
    st.image('Reddit_Photos/EDA_Relationship.png', caption = 'Majority of the submissions with an upvote ratio ranging from 0.90 to 1.0 have less than 10,000 comments', width = 700)

    st.divider()
    
    st.header('Redditors vs their Family and Friends')
    st.markdown('Our findings show that most of the topics that appear in the subreddit relate to their issues with their family or their friends.')
    st.subheader('Main Topics extracted based on Subreddit Titles')
    st.image('Reddit_Photos/EDA_Topic.png',  width=700)
    st.subheader('Top Words Extracted from Subreddit Titles')
    st.image('Reddit_Photos/EDA_title_wordcloud.png')
    st.subheader('Top BIGRAMS Extracted from Subreddit Titles')
    st.image('Reddit_Photos/EDA_title_wordcloud_bigram.png')
    st.subheader('Top TRIGRAMS Extracted from Subreddit Titles')
    st.image('Reddit_Photos/EDA_title_wordcloud_trigram.png')

   
elif my_page == 'How does the Model work?':
    st.title('How does the Model work?')
    st.divider()
    st.header('The Modeling and Predcition Pipeline')
    st.markdown('''Here’s a breakdown of how our modeling pipeline works: First, we split the dataset into a 75% chunk for training and validation, and the remaining 25% for a holdout set. In performing the said split, we maintain a 1:1 ratio between the NTA/YTA data to avoid any bias towards one tag. Within the training and validation set, we use a five-fold stratified KFold cross-validation method. This technique ensures the model gets a good mix of different data slices to train on and test against, improving its robustness.\n\nDuring each fold of cross-validation, we apply the DynamicFewShotGPTClassifier: a model that utilizes a KNN-like algorithm to dynamically select the most similar examples in the prompt to assist in classification. Now this is where the magic happens—we fine-tune parameters like 'n_examples' to maximize accuracy. After running through all the folds, we pick the settings that deliver the highest accuracy across the board. Finally, we put the model to the ultimate test using the holdout set, evaluating its performance on unseen data to ensure it’s ready for real-world challenges. This thorough approach—from rigorous training to meticulous testing—ensures our model is optimized and reliable before it’s put into action.''')
    st.image('Reddit_Photos/Modeling_Pipeline.png')
    st.subheader('Our model obtained a 92% holdout accuracy')
    st.markdown('''After fine-tuning our model through rigorous cross-validation and parameter optimization, we achieved great results during evaluation on the holdout set. Our model demonstrated impressive performance with a holdout accuracy of 92%, correctly identifying 23 out of 25 instances it was sampled to test. This high level of accuracy validates the effectiveness of the model we were able to streamline through our iterations. ''')
    st.image('Reddit_Photos/Modeling_Results.png')
    st.subheader('Prediction: Understanding how ChatGPT gives feedback')
    st.markdown('''In order to come up with the best results, we use the model with the best settings (n_examples=3) to come up with the tagging. Then using the _get_prompt() method of FewShotDynamicGPTClassifier, we can retrieve the training data most similar to the user submission. Besides ChatGPT giving the user input a tagging, we can ask it to generate a prompt asking why it gave that tagging basing their answer on the most similar training data.\n\nBesides ChatGPT giving the user input a tagging, we can also ask it to generate a prompt asking why it gave that tagging basing their answer on the most similar training data.\n\nPrompt: Briefly explain why you gave the {predicted_tagging} to the above user submission. Shortly reference this similar submission with tagging {nearest_comment_tag} as a basis for your explanation: {nearest_comment}''')
    st.image('Reddit_Photos/Modeling_Prediction.png', use_column_width = 'Never')
    st.markdown('''Curious to see how that goes? Head to the \'Check if you\'re the A**hole\' page to get started!''')

elif my_page == 'Check if You\'re the A**hole':
    st.title('Check if You\'re the A**hole')
    user_input=''
    
    # Add text input for the user input
    user_input = st.text_input(
        label='Let us hear you! We\'ll tell you if you\'re in the right path',
        value=''
    )

    result =[]
    if st.button("Judge Me"):
        st.write(f"Your story: {user_input}")
        st.divider()
        prompt_passed, gpt_feedback, similar_submission, similar_submission_tag = predicition_pipeline(query=user_input)

        if similar_submission_tag =='nta':
            st.success('Chin up buddy, you did the right thing!')
        else:
            st.error('Sorry bud, you\'re the a\*\*hole.')

        st.subheader('Let\'s hear what our model has to say:')
        st.markdown(gpt_feedback)
        # prompt_passed 
        st.subheader('You may also view the related story which we think shares the same sentiment:')
        st.markdown(f'<div style="color: black">{similar_submission}</div>', unsafe_allow_html=True)
        if similar_submission_tag =='nta':
            st.success('People also think that the OP in this story is not an a\*\*hole.')
        else:
            st.error('People also think that the OP in this story is an a\*\*hole.')

    # '''Last weekend, my friends invited me to a barbecue, and I offered to bring drinks. I spent a lot of time and money selecting a variety of beverages, including some expensive craft beers that I know the group enjoys. When I arrived, one of the friends, who I know doesn't usually contribute much, immediately started taking the craft beers and sharing them with everyone without acknowledging my effort. I got upset and told him he should have asked first and contributed something himself. The atmosphere got tense, and some friends later said I overreacted. AITAH for getting upset and calling him out?'''
