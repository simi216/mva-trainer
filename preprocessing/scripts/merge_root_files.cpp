#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>


#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TSystem.h>



int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <output_file> <input_files_directory> <tree_name>" << std::endl;
        return 1;
    }
    std::string outputFile = argv[1];
    std::string inputDir = argv[2];
    std::string treeName = argv[3];
    std::vector<std::string> inputFiles;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".root") {
            inputFiles.push_back(entry.path().string());
        }
    }
    if (inputFiles.empty()) {
        std::cerr << "No ROOT files found in the specified directory: " << inputDir << std::endl;
        return 1;
    }
    TChain chain(treeName.c_str());
    int file_count = 0;
    for (const auto& file : inputFiles) {
        std::cout << "Adding file: " << ++file_count << "/" << inputFiles.size() << std::endl;
        chain.Add(file.c_str());
    }
    TFile* output = TFile::Open(outputFile.c_str(), "RECREATE");
    if (!output || output->IsZombie()) {
        std::cerr << "Error creating output file: " << outputFile << std::endl;
        return 1;
    }
    TTree* outputTree = chain.CloneTree(-1, "fast");
    if (!outputTree) {
        std::cerr << "Error cloning tree from chain." << std::endl;
        output->Close();
        return 1;
    }
    output->cd();
    outputTree->Write();
    output->Close();
    std::cout << "Merged " << file_count << " files into " << outputFile << std::endl;
    return 0;
}