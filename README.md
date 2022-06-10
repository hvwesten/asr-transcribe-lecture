# Transcribing lectures with ASR

This is the repository of our ASR project where we try to transcribe 
lectures using the Wav2Vec2 model from HuggingFace. 


### Requirements

You can create a conda environment with the required packages by executing
```bash
conda env create -f env.yml
```
You will also need to install `pyctcdecode` and `kenlm` to run the code for 
creating a language model.

### Data
We require the input data to have the following structure:
```
data/
├─ inputs/
│  ├─ yale_econ251/
│  │  ├─ lectures/
│  │  │   ├─ econ251_01_090309.mp3
│  │  │   ├─ ...
│  │  ├─ lectures-tiny/
│  │  │   ├─ 01-tiny.mp3
│  │  │   ├─ ...
│  │  ├─ transcripts/
│  │  │   ├─ 01.txt
│  │  │   ├─ ...
│  │  ├─ transcripts-tiny/
│  │  │   ├─ 01-tiny.txt
│  │  │   ├─ ...
├─ lm/
│  ├─ custom/
│  │  ├─ 5gram.arpa
│  │  ├─ 5gram_correct.arpa
│  │  ├─ full_text.txt
│  ├─ glue/
│  │  ├─ ...
├─ predictions/
│  ├─ yale_econ251/
│  │  ├─ baseline/
│  │  ├─ wav2vec2-base/
│  │  ├─ wav2vec2-base-100h/
│  │  ├─ wav2vec2-base-960h/

```

### Notebooks

The Jupyter notebooks in the `notebooks/` folder can be run sequentially, 
with `01a` or `01b` indicating different research directions within 
the same step.

