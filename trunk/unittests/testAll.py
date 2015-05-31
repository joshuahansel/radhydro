from unittest import TestSuite, defaultTestLoader, TextTestRunner

## Main function.
#
def main():
   # list of test modules
   test_modules = ['testIntegrationUtilities',
                   'testPureAbsorberProblem',
                   'testPureScatteringProblem',
                   'testDiffusionProblem',
                   'testSSConvergence',
                   'testTransientSource',
                   'testRadTransient',
                   'testRadSpatialConvergence']

   # add all tests modules to suite
   suite = TestSuite()
   for test_module in test_modules:
      suite.addTest(defaultTestLoader.loadTestsFromName(test_module))

   # run suite
   TextTestRunner(verbosity=2).run(suite)

# run main function
if __name__ == "__main__":
    main()
