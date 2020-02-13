# Prototype

An implementation of ProtoPNet from [This Looks Like That: Deep Learning for Interpretable Image Recognition](https://papers.nips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition).

Results are logged to TensorBoard which can be run with:

```bash
tensorboard --log_dir=<output_dir>
```

where <output_dir> is the value of the command line argument with the same name. Then, TensorBoard will give a link which can be used in a web browser to see what's happening to the model in real time.

One possible test:
`python -m pyroclast.eager_run --alg prototype --dataset caltech_birds2011 --resize_data_shape 224 224 --output_dir test_prototype`
