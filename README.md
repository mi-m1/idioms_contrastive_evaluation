# idioms_contrastive_evaluation
Code for LLMs are Turning a Blind Eye to Context: Insights from a Contrastive Dataset for Idiomaticity

## To do:
* zeroshot:
  * llama3
    - [X] meta/meta-llama-3-8b-instruct 
    - [X] meta/meta-llama-3-70b-instruct
    - [ ] meta/meta-llama-3.1-405b-instruct ==> running in progress, afte running run the clean_llama3.py script!
  * mistral?
    - [ ] mistralai/mistral-7b-instruct-v0.1 ==> not working with replicate?
    - [ ] mistralai/mistral-7b-instruct-v0.2
        - [ ] run the literal script
        - [ ] clean literal
        - [ ] clean figurative outputs
   * put all cleaned predictions in a single folder [ ]
* oneshot:
  * gpts
    - [X] gpt35turbo
        - [X] clean
    - [X] gpt4
        - [X] clean
    - [X] gpt4o
        - [X] clean
          
  * llama2
    - [ ] meta/llama-2-7b-chat
        - [ ] ran it
        - [ ] clean it
    - [ ] meta/llama-2-13b-chat
        - [ ] ran it
        - [ ] clean it
    - [ ] meta/llama-2-70b-chat
        - [ ] ran it
        - [ ] clean it
  * llama3
    - [ ] meta/meta-llama-3-8b-instruct
        - [ ] ran it
        - [ ] clean it
    - [ ] meta/meta-llama-3-70b-instruct
        - [ ] ran it
        - [ ] clean it
    - [ ] meta/meta-llama-3.1-405b-instruct
        - [ ] ran it
        - [ ] clean it
  * mistral
* frequency analysis
  * split into 4 (more or less) equal groups
  * the log-log graph
