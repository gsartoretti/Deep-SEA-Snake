# Deep-SEA-Snake
Reinforcement learning on a series elastic actuated snake robot

Database used for offline learning and videos of the results:
[here](https://drive.google.com/drive/folders/0B7U-QvG8MT1qdFhvYzRNVWtjQlk?usp=sharing)

### File list

- CompliantSnake0.py: State of the art compliant controller with 6 windows - used to
  generate offline data.
- CompliantSnake.py: Snake controller running the learned model stored in
  "model_tensorboard"
- Results: Videos and Optitrack logs of trial runs, and scripts to plot the
  results.
- A3C.ipnyb: IPython notebook which runs the A3C algorithm with 6 workers
  on the offline database p_experiments.snake located [here](https://drive.google.com/drive/folders/0B7U-QvG8MT1qdFhvYzRNVWtjQlk?usp=sharing)
### Requirements
- Numpy
- Tensorflow
- matplotlib
If necessary, the file "requirements.txt" contains a pip freeze of the
current working project.

**Note: The module "hebiapi" is currently not publicly available, but it is
only used for sending and recieving low-level joint information to the
robot.
## Contact:

[Guillaume Sartoretti](gsartore@andrew.cmu.edu)

[William Paivine](wjp@andrew.cmu.edu)

[Yunfei Shi](yunfei.shi@connect.polyu.hk)
