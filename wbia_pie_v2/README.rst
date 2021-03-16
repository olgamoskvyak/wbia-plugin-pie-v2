================================================================================
Orientation detection results and evaluation
================================================================================

Quantitative results
---------------------

`Cropped whale shark test set <https://wildbookiarepository.azureedge.net/data/pie_v2.whale_shark_cropped_demo.zip>`_
contains unique shark patterns with 2 to 5 sightings per pattern and a total of 333 unique patterns.
The selecting test subset is challenging as each query image has at most 4 matches.

Accuracy
==========
Accuracy of retrieval 1-vs-all is reported.

+----------------------+---------------+--------------+--------------+--------------+
| Dataset              |    Rank 1     |    Rank 5    |    Rank 10   |    Rank 20   |
+======================+===============+==============+==============+==============+
| Whale shark cropped  |     81.5%     |    89.5%     |    91.7%     |    94.7%     |
+----------------------+---------------+--------------+--------------+--------------+
| Whale Shark          |               |              |              |              |
+----------------------+---------------+--------------+--------------+--------------+
| Snow leopards        |               |              |              |              |
+----------------------+---------------+--------------+--------------+--------------+


Qualitative results
--------------------

Whale Shark cropped
====================

Green is an axis-aligned box, Red is a detected object-aligned box. Yellow side indicates a detected front of the animal.

.. figure:: ../examples/000000000019_0.jpg
   :align: center

.. figure:: ../examples/000000000050_0.jpg
   :align: center

.. figure:: ../examples/000000000104_0.jpg
   :align: center

.. figure:: ../examples/000000000127_0.jpg
   :align: center

.. figure:: ../examples/000000000142_0.jpg
   :align: center

.. figure:: ../examples/000000000182_0.jpg
   :align: center