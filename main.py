import os
import unittest
import yaml
import numpy as np
import laneatt.utils.anchors 

class TestAnchors(unittest.TestCase):
    def test_anchor(self):
        self.assertEqual(laneatt.utils.generate_anchor(0.49295774647887325, 60, 72, 10, (720, 1280)), [[0, 0, 365.07, 0, 0, ]        
                                                                                                        ])

if __name__ == '__main__':
    print('Running test cases')
    print(laneatt.utils.anchors.generate_anchor((0, 0.49295774647887325), 60, 72, 10, (720, 1280)))
    unittest.main()