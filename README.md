## BladeGPT

Have you seen the movie "Blade Runner 2049" ?  
Well, I liked the charater 'Officer K' acted by Ryan Gosling.  
So I want to make a nano-scaled GPT for generate sentences that resemble the way officer K speaks.


### How I feel when I start building a Model

<img src="./imgs/bladememe.jpg" alt="drawing" width="500"/>


### Result
<img src="./imgs/three-days.jpg" alt="drawing" width="500"/>


### Notes
- Number of Batch size should be multiple of 2 with the reason of the way GPU works

### History
- 2024/06/14 
  - Finish skeleton code for GPT 
  - read txt file and tokenize it
  - Install tiktoken
  
- 2024/06/15
  - Make a class for config
  - Aggregate two attetion classes into one class
  - Add some assertation statements
  - Add a generating token code
  



### Reference
- This repo is based on Andrej Karpathy's lecture.