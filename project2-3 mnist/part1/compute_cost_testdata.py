# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:45:41 2019

@author: Han-Hsien.Seah
"""

X = np.array([[ 1. ,89. ,19. ,81. ,99. ,63. ,72. ,78. ,22. ,88. ,18.],
 [ 1. ,31. ,75. ,73. ,25. ,57. , 9. ,10. ,86. , 5. ,75.],
 [ 1. ,46. ,10. ,48. ,54. ,29. ,16. ,93. ,24. ,17. ,85.],
 [ 1. ,74. ,26. ,60. ,30. ,98. , 9. ,79. ,12. ,65. ,82.],
 [ 1. ,77. ,13. ,71. ,99. ,59. ,15. ,94. ,84. ,23. ,66.],
 [ 1. ,11. ,62. ,10. ,65. ,61. ,72. ,43. ,33. ,90. , 5.],
 [ 1. ,26. ,52. ,39. ,67. ,45. ,82. ,12. ,93. ,49. , 9.],
 [ 1. ,25. ,36. ,58. ,81. ,90. ,22. ,69. ,91. ,81. ,32.],
 [ 1. ,43. ,86. ,42. ,28. ,31. ,52. , 2. ,91. ,43. ,61.],
 [ 1. ,34. ,17. ,74. ,79. ,85. ,82. ,25. ,94. ,13. ,58.]])

theta = np.array([[0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
 [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.]])

temp_parameter =  1.0
lambda_factor = 0.0001