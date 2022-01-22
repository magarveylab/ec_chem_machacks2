# EC Numbers from Chemical Reactions (MacHacks2 Project)

## Introduction
Enzymes are proteins which catalyze chemical reactions in living organisms. They are ubiquitous and critical to life. Nearly every chemical process in the body is enzyme-dependent,  including such functions as DNA replication, metabolism, tissue repair, and so on. The study of enzymes is important to help combat enzyme disorders, such as phenylketonuria, as well as to better understand how to keep humans healthy. Moreover, enzymes have broad applications in human medicine, environmental bioremediation, wastewater treatment, cleaning (e.g. laundry detergents), etc. Despite their utility, many enzymes are not well characterized. An enzyme classification system known as the Enzyme Commission number (EC #) has been developed to try to catalogue the known functions of enzymes to assist in this effort. The hierarchical EC number system breaks down the thousands of different enzyme functions according to the type of reactions that they catalyse, where each enzyme can have up to four consecutive numbers in the form 1.2.3.4 - where the first number is the most general classification, and each subsequent number provides more specific information. Read more [here](https://en.wikipedia.org/wiki/Enzyme_Commission_number).

## Task Description
Predicting the EC number associated with a given reaction is an important step in assigning the reaction space of a given enzyme. In this project, the goal is to correctly assign at least the first enzyme commission number to each reaction. 

## Evaluation
The anticipated format of submissions will be an enzyme classification model which takes featurized/tokenized chemical representations and returns a predicted enzyme commission number to the first (or further) level. Submissions will be evaluated using the following guidelines.

#### Performance
The performance of the model on the test (i.e hold-out) dataset. Accuracy, precision, and F1 score must be calculated by the entrants and supplied with valid submissions at judging time.
Accuracy/precision/F1 scores at the first EC number level will be prioritized
Bonus points will be awarded for accurate assignment of subsequent EC numbers.

#### Novelty
This challenge has been approached before in published work. Judges will be looking for interesting approaches to feature selection/engineering, and creative use of transfer learning or different architectures to achieve good enzyme classification results.

#### Explainability
In a limited time frame, state-of-the-art performance is not expected. Preference will be given to submissions which justify the performance of their model, including reasons for the observed performance, and strategies for improvement.


## Dataset Description
### SMILES

SMILES is a standardized text language for the representation of chemical molecules and reactions. Each unique chemical entity will have a corresponding unique SMILES, which can be read by a variety of computational programs, such as the Python package RDKit. Below is an example of the molecule pyruvate, for which the SMILES is `OC(C(C)=O)=O`.

![image](https://user-images.githubusercontent.com/61066192/150393470-5a80666d-f6fd-40a5-9df5-3f1114ab7b21.png)

### Reaction SMILES
In addition to molecules, we can use SMILES to represent entire chemical reactions. The reactant and product of the reaction are separated by the characters `>>`. If there are multiple reactant/product molecules, they are separated by a period (`.`). Below is an example of the decarboxylation of pyruvate into acetaldehyde and carbon dioxide, for which the SMILES is `OC(C(C)=O)=O>>O=C=O.[H]C(C)=O`.

![image](https://user-images.githubusercontent.com/61066192/150394453-1edc278a-84ea-4aa4-adff-3cd343e68831.png)


### Mapped Reaction SMILES

We can add further information to a reaction SMILES by mapping the atoms between reactant and product. In other words, we can assign a unique number that tracks an atom as it undergoes a reaction (so long as the atom is found in both reactant and product). Let’s look at the same pyruvate decarboxylation reaction, for which the mapped reaction SMILES is `[OH:1][C:2]([C:3]([CH3:5])=[O:6])=[O:4]>>[O:1]=[C:2]=[O:4].[H][C:3]([CH3:5])=[O:6]`

![image](https://user-images.githubusercontent.com/61066192/150394477-8b41d6f0-ce17-48c3-ba80-10e52c7f4e06.png)

### RDKit

We can use the Python package RDKit to create objects from SMILES strings. The following code can be used to create an RDKit Mol object using SMILES:

```python
from rdkit import Chem

smiles = “OC(C(C)=O)=O”
mol = Chem.MolFromSmiles(smiles)
display(mol)   # the molecule can be displayed in a Jupyter Notebook
```
Mol objects in RDKit store information regarding a number of properties for both atoms and bonds that may be of interest to you. These properties can be used to create molecular descriptors. Several examples of the properties that can be accessed for any atom, and how to do so, are included below.
```python
from rdkit import Chem
	
smiles = “OC(C(C)=O)=O”
mol = Chem.MolFromSmiles(smiles)

#### Iterate through atoms:

for atom in mol.GetAtoms():
	# Get index (each atom in a molecule has a unique index value)
	idx = atom.GetIdx()

	# Get Atomic Number
	atomic_num = atom.GetAtomicNum()

	# Get Atom Map Number (used for mapped reactions)
	map_num = atom.GetAtomMapNum()
	# Get Bonds
	bonds = atom.GetBonds()

	# Get Degree
	degree = atom.GetDegree()

	# Get Hybridization
	hybridization = atom.GetHybridization()

	# Get Neighbours
	neighbours = atom.GetNeighbors()
```
The full documentation for RDKit atom and bond objects is available [here](https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html)


Additionally, we can use the following code to create a reaction object in RDKit. 

*Note: Reactions in RDKit are created using a similar language called SMARTS. You do not need to worry about SMARTS, but know that you can translate from SMILES to SMARTS (and vice versa) if you need to.*
```python
from rdkit import Chem
from rdkit.Chem import rdChemReactions

reaction_smiles = “OC(C(C)=O)=O>>O=C=O.[H]C(C)=O”

reactant_smiles = reaction_smiles.split(“>>”)[0]
reactants = [Chem.MolFromSmiles(x) for x in reactant_smiles.split(“.”)]
reactant_smarts = “.”.join([Chem.MolToSmarts(x) for x in reactants])

product_smiles = product_smiles.split(“>>”)[1]
products = [Chem.MolFromSmiles(x) for x in product_smiles.split(“.”)]
product_smarts = “.”.join([Chem.MolToSmarts(x) for x in products])

reaction_smarts = “{}>>{}”.format(reactant_smarts, product_smarts)
reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
display(reaction)   # the reaction can be displayed in a Jupyter Notebook
```

## Suggestions & Hints

This project can be divided into two stages:
1. **Featurization of Molecules**: Translating chemical information to numerical formats as inputs to ML algorithms
2. **Supervised classification of reactions**: Training a classifier on the features extracted from the reactants and products of a reaction

The following sections provide examples of how one may pursue different avenues of this project. There are a lot more examples available online. It is recommended that the participants do some research to develop a novel solution.

### Molecular Featurization

#### Treating SMILES as a language and extracting embeddings using natural language processing

In recent years, transformers have become extremely popular in understanding the molecular structures encoded in smiles to predict chemical properties. There are many pretrained models available such as ChemBerta (available on [HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)), which was trained on a large corpus of 100K smiles. Here is an example of how to use this model to extract numerical embeddings from SMILES.
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") 

smiles = “OC(C(C)=O)=O>>O=C=O.[H]C(C)=O”
embedding = tokenizer.encode_plus(smiles)
```

If you are interested in pursuing natural language processing for this project, it is recommended that you take advantage of transfer learning (fine-tuning existing weights in the model for new tasks such reaction classification)

### Fingerprint representation of molecules

Rdkit, a popular chemoinformatics package, has different ways to generate fingerprints for molecules. Here is an example of generating such representation with python:

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

smiles = 'CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O'
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=1024)
vector = list(fp)
```

More detailed information on this technique can be found [here](https://chemicbook.com/2021/03/25/a-beginners-guide-for-understanding-extended-connectivity-fingerprints.html).


### Alternative representation of molecules

DeepChem provides many different ways to featurize molecules for different model architectures such as graph neural networks. The different types of featurizations can be found [here](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html) for your convenience.

**We look forward to seeing your projects!**
