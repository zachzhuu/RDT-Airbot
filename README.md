# Finetuning RDT-1B on Self-Collected Data

This repository is forked from [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) and part of our course project for [AIAA 5032, Spring 2025](https://hkust-aiaa5032.github.io/spring2025/) at HKUST(GZ).

## Installation

Please refer to the original [RDT repository](https://github.com/thu-ml/RoboticsDiffusionTransformer) for detailed installation instructions.

## Data Collection

We used a single-arm [AIRBOT Play](https://airbots.online/) robot to collect our dataset, which consists of 490 episodes of human demonstrations across various tasks. The data collection script can be found at `data/dataset_collect_RDT.py` on the `jiahong` branch.

## Qualitative Results

Below are selected examples from our testing phase.

### Generalization Test

![Generalization](assets/generalize.gif)

(a–b) The model places a seen toy into previously unseen containers with different shapes and colors.  
(c) The model fails to recognize an unseen shuttlecock, resulting in an unsuccessful attempt.  
(d) The model successfully grasps an unseen Christmas tree toy and places it into a known box.

### Long-Horizon Task

![Long-horizon](assets/fail_duck.gif)

In this long-horizon task, the robot successfully picks and places the lid but consistently fails to grasp the rubber duck.

### State-Aware Behavior

![State-aware](assets/state_test.gif)

During the long-horizon task, a human operator manually places the duck into the box to compensate for failed grasps. The model correctly perceives the updated state and skips the duck-picking step, proceeding to the next action—picking up the lid.

## Project Report

For more details, please refer to our [project report](assets/report.pdf).
