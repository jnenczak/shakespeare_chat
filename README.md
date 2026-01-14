# shakespeare_chat
Its small local chatbot that answers questions about William Shakespeare using Wikipedia (RAG) for facts, while sounding more Shakespearean thanks to a light LoRA fine-tune on his plays.

We do two things here:
- We train a small LoRA adapter on `shakespeare.txt`. its all Shakespeare works in plain .txt, its use to change how model sounds not what it says
- Then we use RAG and Wikipedia articale to search information about Shakespeare to anserw question about him

Use chat_shakespeare_rag.py to chat
