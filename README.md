# idioms_contrastive_evaluation
Code for LLMs are Turning a Blind Eye to Context: Insights from a Contrastive Dataset for Idiomaticity

## Evaluator Scripts
0shot evaluator:
https://github.com/mi-m1/idioms_contrastive_evaluation/blob/master/prompting/analysis_for_paper/model_evaluator.py

1shot evaluator:
https://github.com/mi-m1/idioms_contrastive_evaluation/blob/master/prompting/analysis_for_paper/model_evaluator_fewshot.py

## Notes on Metrics
https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
https://www.evidentlyai.com/classification-metrics/multi-class-metrics#visual-example-micro

## Papers for Inspiration
https://link.springer.com/article/10.1007/BF01068157

https://osf.io/c2yef/?view_only=

## To do:
* zeroshot:
  * llama3
    - [X] meta/meta-llama-3-8b-instruct 
    - [X] meta/meta-llama-3-70b-instruct
    - [X] meta/meta-llama-3.1-405b-instruct <s>==> running in progress, afte running run the clean_llama3.py script!</s>
  * mistral?
    - [ ] mistralai/mistral-7b-instruct-v0.1 ==> not working with replicate?
    - [ ] mistralai/mistral-7b-instruct-v0.2
        - [ ] run the literal script
        - [ ] clean literal
        - [ ] clean figurative outputs
    - [ ] sort out mistral!! not saving label, only "output"
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
    - [X] meta/llama-2-7b-chat
        - [X] ran it
        - [X] clean it
    - [X] meta/llama-2-13b-chat
        - [X] ran it
        - [X] clean it
    - [X] meta/llama-2-70b-chat
        - [X] ran it
        - [X] clean it
  * llama3
    - [X] meta/meta-llama-3-8b-instruct
        - [X] ran it
        - [X] clean it
    - [X] meta/meta-llama-3-70b-instruct
        - [X] ran it
        - [X] clean it
    - [X] meta/meta-llama-3.1-405b-instruct
        - [X] ran it
        - [X] clean it
  * flan models
  * mistral
* model evaluator bug
  - [X] calculate f1 scores for the whole dataset
* frequency analysis
  * <s>split into 4 (more or less) equal groups</s>
  * log-log graph based on F1
  * familarity vs. frequency graph
    - [X] Tried with lemma frequency of the whole expression: plot_macro and plot_micro folders
    - [ ] Try with familiarity measures
    - [ ] Try with different frequency measures?
        - [ ] Kucera and Francis
  * familarity vs. performance (F1)



