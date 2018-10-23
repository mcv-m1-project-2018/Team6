# Painting Retrieval


Introduction to Human and Computer Vision project example code and data for the Painting Retrieval project



DATA FILES:
- query_corresp_simple_devel.pkl: True correspondences between query images and museum database images for the development simple_query. The correspondences are stored in a python dictionary where the key is the number in the name of the query image and the value is the number in the name of the dictionary image. This is necessary to evaluate your development results using the provided mapk() function.

  The correspondences:
  {0: 76, 1: 105, 2: 34, 3: 83, 4: 109, 5: 101, 6: 57, 7: 27, 8: 50, 9: 84, 10: 25, 11: 60, 12: 45, 13: 99, 14: 107, 15: 44, 16: 65, 17: 63, 18: 111, 19: 92, 20: 67, 21: 22, 22: 87, 23: 85, 24: 13, 25: 39, 26: 103, 27: 6, 28: 62, 29: 41}

  For instance, for key '0', the value is '76'. This means that, for query image '000000.jpg' the corresponding true correspondence is image '000076.jpg' in the museum database.

  Note that in python you can format a number with trailing zeros by using:

  Python 3.6.2 (default, Mar 13 2018, 08:54:27)

  &gt;&gt;&gt; id = 76

  &gt;&gt;&gt; str = '{:06d}.jpg'.format(id)

  &gt;&gt;&gt; print (str)

  000076.jpg

