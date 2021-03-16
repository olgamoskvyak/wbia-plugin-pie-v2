================================================================================
Orientation detection results and evaluation
================================================================================

Quantitative results
---------------------

`Cropped whale shark test set <https://wildbookiarepository.azureedge.net/data/pie_v2.whale_shark_cropped_demo.zip>`_
has images with between 2 and 5 images per unique shark pattern
(combination name/viepoint) with total 333 unique names.
The selecting test subset is challenging as each query image has at most 4 matches.
Accuracy of retrieval 1-vs-all is reported.

Accuracy
==========

Accuracy of predicting an angle of orientation on **a test set** at **10, 15 and 20 degrees thresholds**:

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

.. figure:: ../examples/whaleshark_bboxes_1.jpg
   :align: center