# Build the docker image 
docker build --tag 'dtd' .

# Run the container
docker run -it --rm --gpus all -v /home/youssef.achenchabe/DAMNETS_ICML_2022:/workspace/DAMNETS_ICML_2022 dtd bash

# activate the environment
conda activate DAM-TAG-DYM

# Generate synthetic data
#### It generates a list of networkx graphs, then a function converts the generated graphs into edgelist format
python generate_data.py -c 3_comm_decay.yaml -d debug_data 


# Run TagGen algorithm on this synthetic dataset
python run_tag_gen.py -p data/debug_data.pkl -o test_taggen/

# Run Dymond algorithm on this synthetic dataset
python run_dymond.py -p data/debug_data.pkl -o test_dymond/


# Run TagGen on real dataset DBLP:
#### create a folder DBLP/train_edgelists/0/edgelist.txt and run the following command
python run_tag_gen.py -p data/DBLP/ -o data/DBLP/ -r True -nbg 1