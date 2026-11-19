# Third-party notices

## CLINC150 / CLINC out-of-scope intent dataset

The experiment uses **An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction**, by Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars (2019).

- [Original repository](https://github.com/clinc/oos-eval)
- [Paper, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1131/)
- [Pinned data source](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json)
- [Upstream license at the same revision](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE): Creative Commons Attribution 3.0 Unported ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)).

Changes made for this experiment: canonical duplicate texts were quarantined across all original splits; the remaining validation examples were deterministically partitioned into distinct development roles; labels were sorted; source identities and checksums were added. Tokenizer vocabulary and model parameters were learned from the cleaned training texts. These changes and derived artifacts are not endorsed by the dataset authors. The source dataset is not redistributed in Git or in the model archive.

The repository's [noncommercial license](LICENSE) covers this project's original code and model artifacts to the extent its copyright holders can grant those rights; commercial use requires a separate written agreement and an agreed fee. Earlier releases retain their original license grants. It does not replace the original data license. Retain this attribution with the released bundle and derived experiment materials.

## Software

JAX, Flax, Optax, Hugging Face Transformers/Tokenizers, scikit-learn, NumPy, SciPy, safetensors, FastAPI, and other packages retain their respective licenses. Dependencies are installed from package repositories; their source or binaries are not vendored here. Exact tested package versions are in `requirements-lock.txt`.
