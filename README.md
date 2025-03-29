# Eraser4RAG
source code for the paper 'Learning to Erase Private Knowledge from Multi-Documents for Retrieval-Augmented Large Language Models'
## Code Structure
```datasets``` contains all helper functions to process datasets (including processing retrieved docs and relation extraction) and the annotated datasets for SFT.\
```utils``` contains all helper functions to post-process the datasets with triplets.\
```data_structure.py``` contains methods for constructing and calling datasets\
```finetune_rewrite_doc.py``` contains the code to SFT the rewritting model.\
```RL_train.py``` contains the code to train the rewritting model using the ppo algorithm.\
```get_special_data.py``` contains the code to select the test data $D_{special}$.\
```get_inference_attack_data.py``` contains the code to select the test data $\{D\}_{infer}$.
## Setup Environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```
## References
The code for retrieval refers to https://github.com/AkariAsai/self-rag.

The code for reinforcement learning refers to https://github.com/huggingface/trl.
