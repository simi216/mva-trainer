#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <TROOT.h>

#include "../include/PreProcessor.h"
#include <TLorentzVector.h>


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <tree_name>" << std::endl;
        return 1;
    }

    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];
    const char* treeName = argv[3];

    if (!std::filesystem::exists(inputFileName)) {
        std::cerr << "Error: Input file does not exist: " << inputFileName << std::endl;
        return 1;
    }
    if (std::filesystem::exists(outputFileName)) {
        std::cerr << "Warning: Output file already exists and will be overwritten: " << outputFileName << std::endl;
    }

    PreProcessor preProcessor(inputFileName, outputFileName, treeName);

    // check for additional flags
    if (argc > 4) {
        std::string flag = argv[4];
        if (flag == "-p") {
            std::vector<std::string> parameters;
            for (int i = 5; i < argc; ++i) {
                if (std::string(argv[i]) == "nu_flows") {
                    preProcessor.RegisterNuFlowResults();
                } else if (std::string(argv[i]) == "initial_parton_info") {
                    preProcessor.RegisterInitialStateInfo();
                } else {
                    parameters.push_back(argv[i]);
                }
            }
        }
        else {
            std::cerr << "Error: Unknown flag: " << flag << std::endl;
            return 1;
        }
    }
    



    preProcessor.Process();
    
    return 0;
}
