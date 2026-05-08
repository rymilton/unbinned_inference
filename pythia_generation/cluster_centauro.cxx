#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>

#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
#include "fastjet/contrib/Centauro.hh"
#include "fastjet/EECambridgePlugin.hh"

#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

#define _USE_MATH_DEFINES
int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " --input <input_file> --output <output_file>" << std::endl;
        return 1;
    }

    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) { // Ensure there's a value after the flag
            args[argv[i]] = argv[i + 1];
        }
    }

    if (args.find("--input") == args.end() || args.find("--output") == args.end()) {
        std::cerr << "Error: Both --input and --output arguments are required." << std::endl;
        return 1;
    }

    std::string inputFileName = args["--input"];
    std::string outputFileName = args["--output"];

    double jet_radius;
    if (args.find("--jet_radius") == args.end())
    {
        jet_radius = 0.8;
        std::cout<<"No jet radius argument found. Jet radius set to "<<jet_radius<<std::endl;
    }
    else
    {
        jet_radius = std::stod(args["--jet_radius"]);
    }
    int max_num_jets;
    if (args.find("--max_num_jets") == args.end())
    {
        max_num_jets = 5;
        std::cout<<"No max number of jets argument found. Max set to "<<max_num_jets<<std::endl;
    }
    else
    {
        max_num_jets = std::stoi(args["--max_num_jets"]);
    }

    // ── Open input ROOT file ───────────────────────────────────────────
    TFile input_file(inputFileName.c_str(), "READ");
    if (input_file.IsZombie()) {
        std::cerr << "Error: Could not open input file " << inputFileName << std::endl;
        return 1;
    }
    TTreeReader reader("particles", &input_file);
    if (reader.GetTree() == nullptr) {
        std::cerr << "Error: Could not find tree 'particles' in input file" << std::endl;
        return 1;
    }

    TTreeReaderArray<double> px(reader, "px");
    TTreeReaderArray<double> py(reader, "py");
    TTreeReaderArray<double> pz(reader, "pz");
    TTreeReaderArray<double> E (reader, "E");

    fastjet::contrib::CentauroPlugin *centauro_plugin = new fastjet::contrib::CentauroPlugin(jet_radius);
    fastjet::JetDefinition jet_def(centauro_plugin);

    // ── Prepare output ROOT file ───────────────────────────────────────
    TFile output_file(outputFileName.c_str(), "RECREATE");
    TTree output_tree("jets", "Centauro jets");

    std::vector<double> jet_pt, jet_eta, jet_phi, jet_E, jet_px, jet_py, jet_pz;
    output_tree.Branch("pt",  &jet_pt);
    output_tree.Branch("eta", &jet_eta);
    output_tree.Branch("phi", &jet_phi);
    output_tree.Branch("E",   &jet_E);
    output_tree.Branch("px",  &jet_px);
    output_tree.Branch("py",  &jet_py);
    output_tree.Branch("pz",  &jet_pz);

    while (reader.Next()) {
        // Convert to PseudoJets
        std::vector<fastjet::PseudoJet> particle_vector;
        size_t num_particles = px.GetSize();
        particle_vector.reserve(num_particles);
        for (size_t j = 0; j < num_particles; j++) {
            if (E[j] > 0) {
                particle_vector.emplace_back(px[j], py[j], pz[j], E[j]);
            }
        }

        // Perform clustering
        fastjet::ClusterSequence clust_seq(particle_vector, jet_def);
        std::vector<fastjet::PseudoJet> jets = sorted_by_E(clust_seq.inclusive_jets(0));

        jet_pt.clear(); jet_eta.clear(); jet_phi.clear(); jet_E.clear();
        jet_px.clear(); jet_py.clear(); jet_pz.clear();

        int n_jets_to_store = std::min(max_num_jets, static_cast<int>(jets.size()));
        for (int i_jet = 0; i_jet < n_jets_to_store; i_jet++) {
            const auto &jet = jets[i_jet];
            jet_pt.push_back(jet.pt());
            jet_eta.push_back(jet.eta());
            jet_phi.push_back(jet.phi());
            jet_E.push_back(jet.E());
            jet_px.push_back(jet.px());
            jet_py.push_back(jet.py());
            jet_pz.push_back(jet.pz());
        }

        output_tree.Fill();
    }

    output_file.cd();
    output_tree.Write();
    output_file.Close();
    input_file.Close();

    delete centauro_plugin;
    return 0;
}
