// main341.cc is a part of the PYTHIA event generator.
// Copyright (C) 2026 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: basic usage; DIS

// Basic setup for Deeply Inelastic Scattering at HERA.

#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TH1D.h"
#include "TTree.h"
#include "LHAPDF/LHAPDF.h"
#include <vector>
using namespace Pythia8;
using namespace LHAPDF;

//==========================================================================

int main(int argc, char* argv[]) {

  // Parse alpha_s from command line argument; default to 0.14 if not provided.
  double alphaSvalue = 0.118;
  if (argc >= 2) {
    std::istringstream ss(argv[1]);
    if (!(ss >> alphaSvalue)) {
      std::cerr << "Error: invalid alpha_s value '" << argv[1] << "'" << std::endl;
      return 1;
    }
  }

  // Parse nEvent from command line argument; default to 10000000 if not provided.
  int nEvent = 10000000;
  if (argc >= 3) {
    std::istringstream ss(argv[2]);
    if (!(ss >> nEvent) || nEvent <= 0) {
      std::cerr << "Error: invalid nEvent value '" << argv[2] << "'" << std::endl;
      return 1;
    }
  }
  // Parse lepton_id from command line argument; default to -11 (positron) if not provided.
  int lepton_id = -11;
  if (argc >= 4) {
    std::istringstream ss(argv[3]);
    if (!(ss >> lepton_id) || (lepton_id != 11 && lepton_id != -11)) {
      std::cerr << "Error: invalid lepton_id value '" << argv[3] << "' (must be 11 or -11)" << std::endl;
      return 1;
    }
  }
  const char* lepton_str = (lepton_id == -11) ? "eplus" : "eminus";

  std::cout << "Running with alpha_s = " << alphaSvalue
            << ", nEvent = " << nEvent
            << ", lepton_id = " << lepton_id << std::endl;

  // Beam energies, minimal Q2, number of events to generate.
  double eProton   = 920.;
  double eElectron = 27.6;
  double Q2min     = 150.;
  const double ymin = 0.2;
  const double ymax = 0.7;

  // Generator. Shorthand for event.
  Pythia pythia;
  Event& event = pythia.event;

  // Set up incoming beams, for frame with unequal beam energies.
  pythia.readString("Beams:frameType = 2");
  // BeamA = proton.
  pythia.readString("Beams:idA = 2212");
  pythia.settings.parm("Beams:eA", eProton);
  // BeamB = electron. -11 is positron, 11 is electron
  // put lepton_id in place of -11 to be able to switch between electron and positron beam
  pythia.readString("Beams:idB = " + std::to_string(lepton_id));
  pythia.settings.parm("Beams:eB", eElectron);

  // Set up DIS process within some phase space.
  // Neutral current (with gamma/Z interference).
  pythia.readString("WeakBosonExchange:ff2ff(t:gmZ) = on");
  // Uncomment to allow charged current.
  //pythia.readString("WeakBosonExchange:ff2ff(t:W) = on");
  // Phase-space cut: minimal Q2 of process.
  pythia.settings.parm("PhaseSpace:Q2Min", Q2min);

  // Set dipole recoil on. Necessary for DIS + shower.
  pythia.readString("SpaceShower:dipoleRecoil = on");

  // Allow emissions up to the kinematical limit,
  // since rate known to match well to matrix elements everywhere.
  pythia.readString("PartonShowers:model     = 1" ); // This is the default, but explicitly set here for clarity.
  pythia.readString("SpaceShower:pTmaxMatch = 2"); // Default is 0. Set to 2 in example 341
  pythia.readString("TimeShower:pTmaxMatch   = 1" ); // This is the default, but explicitly set here for clarity.

  // QED radiation off lepton not handled yet by the new procedure.
  pythia.readString("PDF:lepton = off"); // This is ISR. Default is on. Set to off in example 341
  pythia.readString("TimeShower:QEDshowerByL = off"); // Default is on. Set to off in example 341

  pythia.readString("PDF:pSet = LHAPDF6:NNPDF31_nnlo_as_0118");
  pythia.readString("PDF:pHardSet = LHAPDF6:NNPDF31_nnlo_as_0118"); // Defaults to whatever pSet is set to, but explicitly set here for clarity.
  pythia.readString("PDF:useHard  = off");  // This is the default, but explicitly set here for clarity.
  pythia.settings.parm("SigmaProcess:alphaSvalue", alphaSvalue);
  pythia.readString("SigmaProcess:alphaSorder = 1");
  pythia.settings.parm("TimeShower:alphaSvalue", alphaSvalue);
  pythia.readString("TimeShower:alphaSorder = 1");
  pythia.settings.parm("SpaceShower:alphaSvalue", alphaSvalue);
  pythia.readString("SpaceShower:alphaSorder = 1");

  // hadron-level on/off. By default this is on and it seems like it should be on but Vinny has it turned off
  pythia.readString("HadronLevel:Hadronize = on"); 

  pythia.readString("PartonLevel:FSR = on"); // This is the default, but explicitly set here for clarity.

  pythia.readString("ParticleDecays:limitTau0 = off"); // When on, only particles with tau0 < tau0Max are decayed. default is off. This is the only diference between mine and Vinny's generation
  pythia.readString("ParticleDecays:tau0Max = 10"); // Default is 10, but explicitly set here for clarity.

  // Suppress Pythia event/progress printouts.
  pythia.readString("Next:numberCount = 0");


  // If Pythia fails to initialize, exit with error.
  if (!pythia.init()) return 1;

  // ROOT output file.
  TString outPath = TString::Format("/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_files/pythia_H1_alphaS%.4f_%s_%devents.root", alphaSvalue, lepton_str, nEvent);
  TFile* outFile = new TFile(outPath, "RECREATE");

  // ROOT histograms.  
  // TTree for event-level data.
  double tQ2, tW, tx, ty, tWeight;
  TTree* tree = new TTree("events", "DIS event kinematics");
  tree->Branch("Q2",  &tQ2,  "Q2/D");
  tree->Branch("W",   &tW,   "W/D");
  tree->Branch("x",   &tx,   "x/D");
  tree->Branch("y",   &ty,   "y/D");
  tree->Branch("weight", &tWeight, "weight/D");


  // ROOT histograms.  
  // TTree for scattered electron.
  double electron_px, electron_py, electron_pz, electron_e, electron_pt, electron_eta, electron_phi;
  TTree* electron_tree = new TTree("electron", "Scattered electron kinematics");
  electron_tree->Branch("px", &electron_px, "px/D");
  electron_tree->Branch("py", &electron_py, "py/D");
  electron_tree->Branch("pz", &electron_pz, "pz/D");
  electron_tree->Branch("e",  &electron_e,  "e/D");
  electron_tree->Branch("pt",  &electron_pt,  "pt/D");
  electron_tree->Branch("eta",  &electron_eta,  "eta/D");
  electron_tree->Branch("phi",  &electron_phi,  "phi/D");


  // TTree for full particle event record.
  std::vector<int>    pid, status, mother1, mother2, daughter1, daughter2;
  std::vector<double> px, py, pz, e, pt, eta, phi;
  TTree* ptree = new TTree("particles", "Selected visible final-state particles excluding scattered lepton");
  ptree->Branch("pid",       &pid);
  ptree->Branch("status",    &status);
  ptree->Branch("mother1",   &mother1);
  ptree->Branch("mother2",   &mother2);
  ptree->Branch("daughter1", &daughter1);
  ptree->Branch("daughter2", &daughter2);
  ptree->Branch("px",        &px);
  ptree->Branch("py",        &py);
  ptree->Branch("pz",        &pz);
  ptree->Branch("e",         &e);
  ptree->Branch("pt",        &pt);
  ptree->Branch("eta",       &eta);
  ptree->Branch("phi",       &phi);


  // Begin event loop.
  int num_accepted_events = 0;
  while(num_accepted_events < nEvent) {
    if (!pythia.next()) continue;
    if(num_accepted_events % 10000 == 0) {
      std::cout << "Generated " << num_accepted_events << " events." << std::endl;
    }


    // Event kinematics.
    tQ2  = pythia.info.Q2DIS();
    tW   = pythia.info.WDIS();
    tx   = pythia.info.xDIS();
    ty   = pythia.info.yDIS();
    tWeight = pythia.info.weight();

    if (ty < ymin || ty > ymax || tQ2 < 150) continue;
    

    // Fill full particle record.
    pid.clear();       status.clear();
    mother1.clear();   mother2.clear();
    daughter1.clear(); daughter2.clear();
    px.clear(); py.clear(); pz.clear(); e.clear();
    pt.clear(); eta.clear(); phi.clear();

    bool found_scattered_electron = false;
    int scattered_lepton_index = -1; // Index of scattered lepton in event record
    // Finding the scattered lepton
    for (int i = 0; i < event.size(); ++i) {
      if (event[i].id() == lepton_id && event[i].isFinal() && event[i].isVisible() && (event[i].status() < 91 || event[i].status() > 99)) 
      {
          Vec4 particle_4vector = event[i].p();
          double part_pt = particle_4vector.pT();
          double part_eta = particle_4vector.eta();
          double part_phi = particle_4vector.phi();
          double part_px = particle_4vector.px();
          double part_py = particle_4vector.py();
          double part_pz = particle_4vector.pz();
          double part_energy = particle_4vector.e();
  
          // If we already found the scattered electron, compare the energy and pT to the old one to make sure the stored scattered electron has the highest energy and pT
          if (!found_scattered_electron) 
          {
            electron_px = part_px;
            electron_py = part_py;
            electron_pz = part_pz;
            electron_e  = part_energy;
            electron_pt = part_pt;
            electron_eta = part_eta;
            electron_phi = part_phi;
            found_scattered_electron = true;
            scattered_lepton_index = i;
          }
          else 
          {
            double old_electron_pt = electron_pt;
            // New electron must have higher energy and pT than old electron to be considered the scattered electron
            if (part_pt > old_electron_pt) 
            {
              electron_px = part_px;
              electron_py = part_py;
              electron_pz = part_pz;
              electron_e  = part_energy;
              electron_pt = part_pt;
              electron_eta = part_eta;
              electron_phi = part_phi;
              scattered_lepton_index = i;
            }
          }
      }
    }
    if (!found_scattered_electron) continue;

    // Now saving the other particles, but skipping the scattered lepton
    for (int i = 0; i < event.size(); ++i) 
    {
      if (i == scattered_lepton_index) continue; // Skip the scattered lepton, which we've already stored in the electron tree
      if (event[i].isFinal() == false) continue; // Only store final state particles, which are the ones we will cluster into jets.
      if (event[i].isVisible() == false) continue; // Only store visible particles (particles that hit detectors)
      if (event[i].idAbs() == 12 || event[i].idAbs() == 14 || event[i].idAbs() == 16) continue; // Don't store neutrinos, which are invisible and won't hit detectors
      Vec4 particle_4vector = event[i].p();
      double part_pt = particle_4vector.pT();
      double part_eta = particle_4vector.eta();
      double part_phi = particle_4vector.phi();
      double part_px = particle_4vector.px();
      double part_py = particle_4vector.py();
      double part_pz = particle_4vector.pz();
      double part_energy = particle_4vector.e();
      // Apply particle cuts -- eta > -1.5 and eta < 2.75 and pT>0.1
      if (part_eta < -1.5 || part_eta > 2.75 || part_pt < 0.1 ) continue;
      pid.push_back(       event[i].id() );
      status.push_back(    event[i].status() );
      mother1.push_back(   event[i].mother1() );
      mother2.push_back(   event[i].mother2() );
      daughter1.push_back( event[i].daughter1() );
      daughter2.push_back( event[i].daughter2() );
      px.push_back( part_px );
      py.push_back( part_py );
      pz.push_back( part_pz );
      e.push_back(  part_energy );
      pt.push_back( part_pt );
      eta.push_back( part_eta );
      phi.push_back( part_phi );
    }
      
    // If there are no final state particles that satisfy cuts, skip event
    if (pid.size() == 0) continue;
    
    num_accepted_events++;


    tree->Fill();
    electron_tree->Fill();
    ptree->Fill();

  // End of event loop. Statistics.
  }
  pythia.stat();

  std::cout<<"Saving file"<<std::endl;

  // Write and close ROOT file.
  outFile->Write();
  outFile->Close();
  delete outFile;

  // Done.
  return 0;
}
