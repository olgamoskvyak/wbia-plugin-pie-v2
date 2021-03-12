===============================
Wildbook IA - wbia_pie_v2
===============================

Pose Invariant Embedding Re-identification Plug-in - Part of the WildMe / Wildbook IA Project.

A plugin for re-identification of wildlife individuals based on unique natural body
markings.

Installation
------------

.. code:: bash

    ./run_developer_setup.sh

REST API
--------

With the plugin installed, register the module name with the `WBIAControl.py` file
in the wbia repository located at `wbia/wbia/control/WBIAControl.py`.  Register
the module by adding the string `wbia_plugin_orientation` to the
list `AUTOLOAD_PLUGIN_MODNAMES`.

Then, load the web-based WBIA IA service and open the URL that is registered with
the `@register_api decorator`.

.. code:: bash

    cd ~/code/wbia/
    python dev.py --web


Python API - UPDATE
----------

.. code:: bash

    python
    >>> import wbia
    >>> import wbia_orientation
    >>> species = 'spotteddolphin'
    >>> url = 'https://cthulhu.dyn.wildme.io/public/datasets/orientation.spotteddolphin.coco.tar.gz'
    >>> ibs = wbia_orientation._plugin.wbia_orientation_test_ibs(species, dataset_url=url)
    >>> aid_list = ibs.get_valid_aids()
    >>> aid_list = aid_list[:10]
    >>> output, theta = ibs.wbia_plugin_detect_oriented_box(aid_list, species, False, False)
    >>> expected_theta = [-0.4158303737640381, 1.5231519937515259,
                          2.0344438552856445, 1.6124389171600342,
                          1.5768203735351562, 4.669830322265625,
                          1.3162155151367188, 1.2578175067901611,
                          0.9936041831970215,  0.8561460971832275]
    >>> import numpy as np
    >>> diff = np.abs(np.array(theta) - np.array(expected_theta))
    >>> assert diff.all() < 1e-6

The function from the plugin is automatically added as a method to the ibs object
as `ibs.wbia_plugin_detect_oriented_box()`, which is registered using the
`@register_ibs_method decorator`.

Code Style and Development Guidelines
-------------------------------------

Contributing
~~~~~~~~~~~~

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any commit you make. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code:: bash

    git add .
    pre-commit run

    # or

    pre-commit run --all-files

Brunette
~~~~~~~~

Our code base has been formatted by Brunette, which is a fork and more configurable version of Black (https://black.readthedocs.io/en/stable/).

Flake8
~~~~~~

Try to conform to PEP8.  You should set up your preferred editor to use flake8 as its Python linter, but pre-commit will ensure compliance before a git commit is completed.

To run flake8 from the command line use:

.. code:: bash

    flake8


This will use the flake8 configuration within ``setup.cfg``,
which ignores several errors and stylistic considerations.
See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.

PyTest
~~~~~~

Our code uses Google-style documentation tests (doctests) that uses pytest and xdoctest to enable full support.  To run the tests from the command line use:

.. code:: bash

    pytest

To run doctests with `+REQUIRES(--web-tests)` do:

.. code:: bash

    pytest --web-tests

Results and Examples
---------------------

Quantitative and qualitative results are presented `here </wbia_orientation>`_


Implementation details
----------------------
Dependencies
~~~~~~~~~~~~~
* Python >= 3.7
* PyTorch >= 1.5
* Torchvision =- 0.8

Source Data - TODO add links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data used for training and evaluation:

 * whale shark (whole body) - `orientation.seaturtle.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.seaturtle.coco.tar.gz>`_
 * whale shark (cropped) - `orientation.seadragon.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.seadragon.coco.tar.gz>`_
 * show leopards

Key annotations required:

* bounding box containing a region of interest
* name of an animal
* viewpoint (left or right side for sharks)

Viewpoint is an important parameter as left and right sides are different and
are considered as different identities for re-identification purposes.
Identity label is a concatenation of name and viewpoint.


Train/test split and evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Source data is not split into train/test subsets.
The proposed split is based on the number of images per identity.

Train split consists of all identities with more than 6 images per identity.
The rest is moved to the test set.
Note that identities in train and test sets are disjoint.

Re-identification performance is evaluated as one-vs-all retrieval on test set.
Test set is challenging as each identity has a small number of matching images(no more than 6).
There are 335 individuals with only 1 image.
These images act as distractors during evaluation.



Data preprocessing
~~~~~~~~~~~~~~~~~~

Each dataset is preprocessed to speed-up image loading during training. At the first time of running a training or a testing script on a dataset the following operations are applied:
 * an object is cropped based on a segmentation boudnding box from annotations with a padding around equal to the half size of the box to allow for image augmentations
 * an image is resized so the smaller side is equal to the double size of a model input; the aspect ratio is preserved.

The preprocessed dataset is saved in `data` directory.

Data augmentations
~~~~~~~~~~~~~~~~~~

During the training the data is augmented online in the following way:

 * Random Horizontal Flips
 * Random Vertical Flips
 * Random Rotations
 * Random Scale
 * Random Crop
 * Color Jitter (variations in brightness, hue, contrast and saturation)

Both training and testing data are resized to the model input size and normalized.

Training
~~~~~~~~~~~~

Run the training script:

.. code:: bash

  python wbia_orientation/train.py --cfg <path_to_config_file> <additional_optional_params>

Configuration files are listed in `experiments` folder. For example, the following line trains the model with parameters specified in the config file:

.. code:: bash

  python wbia_orientation/train.py --cfg wbia_orientation/config/mantaray.yaml


To override a parameter in config, add this parameter as a command line argument:

.. code:: bash

  python wbia_orientation/train.py --cfg wbia_orientation/config/mantaray.yaml TRAIN.BS 64

Testing
~~~~~~~~~~~~

The test script evaluates on the test set with the best model saved during training:

.. code:: bash

  python wbia_orientation/test.py --cfg <path_to_config_file> <additional_optional_params>

For example:

.. code:: bash

  python wbia_orientation/test.py --cfg wbia_orientation/config/mantaray.yaml

By default, the accuracy of detected rotation angle is computed for a threshold of 10 degrees.
Pass a different value as a command line parameter to evaluate with another threshold:

.. code:: bash

  python wbia_orientation/test.py --cfg wbia_orientation/config/mantaray.yaml TEST.THETA_THR 15.de
