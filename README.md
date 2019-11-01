# Novelty Detection using an adversarial training scheme (implemented using tf.keras)
Detecting unknown objects in semantic segmentations is crucial for detecting corner cases in autonomous driving. This problem is far from solved. We propose a novel architecture that implicitly yields novelties.

The repository is organized in the following way. The **model** directory contains our architecture (subsumed in *NoveltyGAN.py*), the **trainers** directory contains our training schedule. 
Training hyperparameters are set in the .json files in the **configs** directory. Training can be started using **python mains/main.py**
