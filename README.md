# ConMask

Code for AAAI'18 paper: [Open-world Knowledge Graph Completion](https://arxiv.org/abs/1711.03438).

The datasets and pre-trained models can be found at https://drive.google.com/open?id=1YBKw4nOnbscpDeTD_gWxfcpHRFG3MY20

**Warning: Current implementation needs a machine with four GPUs, this could be reduced to 1 GPU but needs code modification.**

## Run the code

### Compile C++ Operator

Go to `ndkgc/ops/__sampling`, run

```bash
cmake .
cmake --build .
```

to compile the negative sampling operator using CMake and TensorFlow.

**If you can not compile this C++ operator, please consider downgrade your TensorFlow to 1.3. PRs to fix this is welcomed.**

### Download per-trained model snapshot

Download DB50 from 

https://drive.google.com/file/d/1qw8d0LGT18D_3p2_dNmyqztkgZO4ageW/view?usp=sharing

put the DB50 dataset under `ConMask/data`.

Download pre-trained ConMask model from

https://drive.google.com/file/d/1OsSwP2LTHiPzP_gManIrjdAxUjj9nl8t/view?usp=sharing

put the snapshot directly under `ConMask/`

And use the following command under `ConMask/`:

```bash
# Closed-World Evaluation
python3 -m ndkgc.models.fcn_model_v2 checkpoint_db50_v2_dr_uniweight_2 data/dbpedia50 --force_eval --layer 3 --conv 2 --lr 1e-2 --keep_prob 0.5 --max_content 512 --pos 1 --neg 4 --noopen --neval 5000 --eval --nofilter
# Open-World Evaluation
python3 -m ndkgc.models.fcn_model_v2 checkpoint_db50_v2_dr_uniweight_2 data/dbpedia50 --force_eval --layer 3 --conv 2 --lr 1e-2 --keep_prob 0.5 --max_content 512 --pos 1 --neg 4 --open --neval 5000 --eval --filter
``` 

You can also find the DB500 dataset at 
https://drive.google.com/file/d/1Tx1gyMoj-9RkbdRvKzHYZ5EZmSrUywVF/view?usp=sharing

Please submit a GitHub issue if you have any further questions or inquiry baoxu.shi@gmail.com.
