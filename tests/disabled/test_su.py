import numpy as np
import unittest
import pyseis_io.su as su  # Import the su module

class ContainerTests(unittest.TestCase):
    
    def setUp(self):
        self.path = "../data/Line_001_2.su"
        self.su = su.SU(self.path)  # Create an instance of SU

    @unittest.skip("Requires external data file")
    def test_initialise_SU_file(self):
        # Your test code here, using self.su
        # Example: self.su.initialize()
        pass

# More test methods as needed
