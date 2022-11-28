# OCR_transformer_persian
Fine-tuning a OCR Transformer Model On Persian Language

A recent Johns Hopkins study claims more than 250,000 people in the U.S. die every year from medical errors. Other reports claim the numbers to be as high as 440,000.This repository is part of a big project having two main prespective:

1) Using state-of-the-art OCR Models: Transformers, Tesseract and PaddlePAddle to read the Clinick discriptions.
2) Using the Multilingual-OCR trained checkpoints for other usage such as CV parsing.

I am affraid I cant provide our latest developed codes and priscriptions dataset right now since it is a comercial verison. In the near future we will publish our latest updates and webservices.

Also Docer Files will be available. 

I have used Models Provided by HuggingFace so if you want to change your basic model you can go to HuggingFace website and change the Model based on your perefrences. Also the requirements to train and fine-tune transformers models are provided in Huggingface website.

Also The WER I achieved by fine-tuning this model is 75%. This can not be accaptable for prescription reading Since it negetivly impacts on people's lives. Now we try to use other Models such as Tesseract 5 and PaddlePaddle to fine-tune a model with a high accuracy. Also we are trying to create a large dataset based on multilingual Prescription data. 


