## Installation of Dependencies
In the root directory of this project run `pipenv install`, this will install all of the dependencies that the project needs and will create the virtual environment.  If you don't have pipenv you can download it [here](https://github.com/pypa/pipenv).   


#### Downloading the model
##### The script way
Run this command in the root of the directory and it will download the model and extract it correctly.
```bash
wget -qO- https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip | bsdtar -xvf-
```
###### Long way
Once we have this the bert model that we want to use needs to be downloaded and extracted into a directory.  Since we're working on a mix of languages we're going to use the bert mixed model which can be downloaded [here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip).  To make things easier, I'd recommend moving that downloaded zip to the root of the project directory and unzipping it there.  This will unzip into a folder named `multi_cased_L-12_H-768_A-12/` which contains all the files for the pretrained model.

## Running the code
To be able to use our dependencies in the virtual environment you'll need to run `pipenv shell` to get a session within it.  To make our word embeddings easier we're using [bert as a service](https://github.com/hanxiao/bert-as-service) which needs to be started in a separate shell with `bert-serving-start -model_dir ./multi_cased_L-12_H-768_A-12/ -num_worker=4 ` 

Once the bert as a service server is running we can run our code, so get a new virtual environment session in a new shell and from the root directory run `python semeval/main.py` and everything should start working