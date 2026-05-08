"""
Read Pythia DIS events from pythia_H1_output.root, cluster into jets
using FastJet kt algorithm (R=1.0, pT > 10 GeV), compute jet observables,
and save to a new ROOT file.
"""

import uproot
import numpy as np
import fastjet
import awkward as ak
import argparse

# ── Configuration ──────────────────────────────────────────────────────
R = 1.0
PROTON_BEAM = np.array([0.0, 0.0, 920.0, 920.0])  # px, py, pz, E

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cluster jets from Pythia DIS events and compute observables."
    )

    parser.add_argument(
        "--input",
        default="pythia_H1_alphaS1136_eplus_10mil.root",
        help="Full path to input ROOT file"
    )

    parser.add_argument(
        "--output",
        default="pythia_H1_alphaS1136jets_eplus_10mil.root",
        help="Full path to output ROOT file"
    )

    parser.add_argument(
        "-R",
        type=float,
        default=1.0,
        help="Jet radius parameter"
    )

    parser.add_argument(
        "--ptmin",
        type=float,
        default=10.0,
        help="Minimum jet pT (lab frame)"
    )

    parser.add_argument(
        "--breit_ptmin",
        type=float,
        default=5.0,
        help="Minimum jet pT (Breit frame)"
    )
    parser.add_argument(
        "--use_breit",
        action="store_true",
        default=False,
        help="Enable to boost and cluster in the Breit frame (in addition to lab)"
    )
    parser.add_argument(
        "--start_event",
        default=None,
        type=int,
        help="Event number to start with"
    )
    parser.add_argument(
        "--end_event",
        default=None,
        type=int,
        help="Event number to end with"
    )
    parser.add_argument(
        "--use_centauro",
        action="store_true",
        default=False,
        help="Enable to boost and cluster in the Breit frame (in addition to lab)"
    )
    parser.add_argument(
        "--centauro_zjetmin",
        type=float,
        default=0.2,
        help="Minimum zjet value for Centauro jets"
    )
    parser.add_argument(
        "--lepton_beam",
        type=int,
        default=11,
        help="PDG number of lepton beam (11 or -11)"
    )

    return parser.parse_args()


def calculate_q(hfs_pz, hfs_energy, elec_px, elec_py, elec_pz, elec_E):
    """Compute the virtual photon 4-momentum q = k - k' using the sigma method."""
    sigma_h = np.sum(hfs_energy - hfs_pz) # sum over hadrons: E - pz
    elec_p = np.sqrt(elec_px**2 + elec_py**2 + elec_pz**2)
    elec_theta = np.arccos(elec_pz / elec_p) if elec_p > 0 else 0.0
    sigma_eprime = elec_E * (1 - np.cos(elec_theta))
    sigma_tot = sigma_h + sigma_eprime


    # Reconstruct beam electron from sigma method
    beam_px, beam_py = 0.0, 0.0
    beam_pz = -sigma_tot / 2.0
    beam_E = sigma_tot / 2.0

    q = np.array([beam_px - elec_px,
                  beam_py - elec_py,
                  beam_pz - elec_pz,
                  beam_E  - elec_E])
    return q

def boost_particles_to_breit(px, py, pz, E, q):
    """
    Vectorized Breit-frame boost.
    Input arrays are 1D for one event.
    Returns boosted px, py, pz, E arrays.
    """
    qx, qy, qz, qE = q
    qpt = np.sqrt(qx * qx + qy * qy)

    qdotq = qE * qE - qx * qx - qy * qy - qz * qz
    Q = np.sqrt(-qdotq)
    Sigma = qE - qz

    # Boosting the particles to the Breit frame
    # Calculating boost vector
    # Calculation taken from https://doi.org/10.1140/epjc/s10052-024-13003-1

    xboost_px = qx / qpt
    xboost_py = qy / qpt
    xboost_pz = qpt / Sigma
    xboost_E = qpt / Sigma

    yboost_px = -qy / qpt
    yboost_py = qx / qpt

    # boostvector = q - Q^2/Sigma * (-z_hat)
    # with z_hat=(0,0,1,1), so:
    # boostvector = (qx, qy, qz + Q^2/Sigma, qE + Q^2/Sigma)
    Q2_over_sigma = (Q * Q) / Sigma
    boostvector_px = qx
    boostvector_py = qy
    boostvector_pz = qz + Q2_over_sigma
    boostvector_E = qE + Q2_over_sigma

    # Minkowski dot: a·b = aE*bE - ax*bx - ay*by - az*bz
    boosted_px = xboost_E * E - xboost_px * px - xboost_py * py - xboost_pz * pz
    boosted_py = -yboost_px * px - yboost_py * py
    boosted_pz = (qE * E - qx * px - qy * py - qz * pz) / Q
    boosted_E = (boostvector_E * E - boostvector_px * px - boostvector_py * py - boostvector_pz * pz) / Q

    return boosted_px, boosted_py, boosted_pz, boosted_E

# jet_type is either fastjet or numpy, which dictates whether .E(), etc works
def calculate_breit_zjet(breit_jet, Q, jet_type="fastjet"):
    n = np.array([0, 0, 1, 1], dtype=np.float32)
    numerator = n[3] * breit_jet.E() - n[0] * breit_jet.px() - n[1]*breit_jet.py() - n[2]*breit_jet.pz()
    return numerator/Q

def calculate_jet_features(jet, q, P_dot_q):
    """Compute substructure observables for a single jet."""
    tau_10 = 0.0
    tau_11p5 = 0.0
    tau_12 = 0.0
    tau_20 = 0.0
    sumpt = 0.0

    for constituent in jet.constituents():
        dr = jet.delta_R(constituent)
        pt = constituent.pt()
        tau_10 += pt * dr
        tau_11p5 += pt * dr**1.5
        tau_12 += pt * dr**2
        tau_20 += pt**2
        sumpt += pt

    jet_pt = jet.pt()
    log_tau_10 = np.log(tau_10 / jet_pt) if jet_pt > 0 and tau_10 > 0 else np.nan

    # z_jet = (P · jet) / (P · q)
    P_dot_jet = PROTON_BEAM[3]*jet.E() - PROTON_BEAM[0]*jet.px() - PROTON_BEAM[1]*jet.py() - PROTON_BEAM[2]*jet.pz()
    zjet = P_dot_jet / P_dot_q if P_dot_q != 0 else np.nan

    phi = (jet.phi() + np.pi) % (2 * np.pi) - np.pi
    return jet_pt, phi, log_tau_10, zjet


def make_pseudojets(px, py, pz, E):
    return [fastjet.PseudoJet(px[i], py[i], pz[i], E[i]) for i in range(len(px))]

def main():
    flags = parse_arguments()

    # ── Read input ─────────────────────────────────────────────────────
    print(f"Opening file {flags.input}")
    with uproot.open(flags.input) as f:
        particles = f["particles"].arrays(library="np", entry_start = flags.start_event, entry_stop=flags.end_event)
        events = f["events"].arrays(library="np", entry_start = flags.start_event, entry_stop=flags.end_event)
        electrons = f["electron"].arrays(library="np", entry_start = flags.start_event, entry_stop=flags.end_event)

        px = particles["px"]
        py = particles["py"]
        pz = particles["pz"]
        energy = particles["e"]

        Q2 = events["Q2"]
        W = events["W"]
        x = events["x"]
        y = events["y"]

    n_events = len(Q2)

    print(f"Read {n_events} events from {flags.input}")

    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R)

    # ── Output lists (one entry per jet, across all events) ─────────────
    jet_output = {
        "jet_pt": [],
        "jet_tau10": [],
        "zjet": [],
        "deltaphi": [],
    }
    if flags.use_breit:
        jet_output["jet_breit_pt"] = []
        jet_output["zjet_breit"] = []
    if flags.use_centauro:
        breit_particles_for_centauro = {"px": [], "py": [], "pz": [], "E": []}
    unclear_electron_events = 0
    # ── Event loop ─────────────────────────────────────────────────────
    for i in range(n_events):
        if i % 100000 == 0:
            print(f"  Processing event {i}/{n_events}")
        
        elec_px = electrons["px"][i]
        elec_py = electrons["py"][i]
        elec_pz = electrons["pz"][i]
        elec_energy = electrons["e"][i]
        elec_phi = electrons["phi"][i]
        
        hfs_px = px[i]
        hfs_py = py[i]
        hfs_pz = pz[i]
        hfs_energy = energy[i]

        # Compute q for this event
        q = calculate_q(hfs_pz, hfs_energy, elec_px, elec_py, elec_pz, elec_energy)

        # Build PseudoJet list
        lab_pseudojets = make_pseudojets(hfs_px, hfs_py, hfs_pz, hfs_energy)
       
        # Cluster in lab frame
        lab_cluster = fastjet.ClusterSequence(lab_pseudojets, jetdef)
        lab_jets = fastjet.sorted_by_pt(lab_cluster.inclusive_jets(ptmin=flags.ptmin))

        event_jet_pt = []
        event_jet_tau10 = []
        event_zjet = []
        event_deltaphi = []

        P_dot_q = PROTON_BEAM[3] * q[3] - PROTON_BEAM[2] * q[2]
        for jet in lab_jets:
            feats = calculate_jet_features(jet, q, P_dot_q)
            delta_phi = np.abs(np.pi + feats[1] - elec_phi)
            if delta_phi > 2*np.pi:
                delta_phi -= 2*np.pi
            event_jet_pt.append(feats[0])
            event_deltaphi.append(delta_phi)
            event_jet_tau10.append(feats[2])
            event_zjet.append(feats[3])

        jet_output["jet_pt"].append(event_jet_pt)
        jet_output["jet_tau10"].append(event_jet_tau10)
        jet_output["zjet"].append(event_zjet)
        jet_output["deltaphi"].append(event_deltaphi)


        if flags.use_breit or flags.use_centauro:
            # Boost to Breit frame
            breit_px, breit_py, breit_pz, breit_energy = boost_particles_to_breit(hfs_px, hfs_py, hfs_pz, hfs_energy, q)
            if flags.use_breit:
                pseudojets_breit = make_pseudojets(breit_px, breit_py, breit_pz, breit_energy)
                breit_cluster = fastjet.ClusterSequence(pseudojets_breit, jetdef)
                breit_jets = fastjet.sorted_by_pt(breit_cluster.inclusive_jets(ptmin=flags.breit_ptmin))
                Q = np.sqrt(Q2[i])

                jet_output["jet_breit_pt"].append([
                    jet.pt() for jet in breit_jets
                ])
                jet_output["zjet_breit"].append([
                    calculate_breit_zjet(jet, Q) for jet in breit_jets
                ])
            # If we're using Centauro, then save the Breit particles to an output root file so we can do the clustering
            if flags.use_centauro:
                breit_particles_for_centauro["px"].append(breit_px)
                breit_particles_for_centauro["py"].append(breit_py)
                breit_particles_for_centauro["pz"].append(breit_pz)
                breit_particles_for_centauro["E"].append(breit_energy)
    
    if flags.use_centauro:
        import subprocess
        print("Saving Breit particles for Centauro")
        centauro_input_file = "./breit_particles_for_centauro.root"
        centauro_output_file = "./centauro_jets.root"
        print("Clustering jets using Centauro")
        with uproot.recreate("./breit_particles_for_centauro.root") as fout:
            fout["particles"] = breit_particles_for_centauro
        subprocess.run(
            [f"source ./run_centauro.sh --input {centauro_input_file} --output {centauro_output_file} --max_num_jets {40} --jet_radius {1.0}"], shell=True
        )
        subprocess.run([f"rm {centauro_input_file}"], shell=True)

        # Now we need to open the Centauro file and calculate zjet, jet pT, and Delta z
        print("Opening Centauro file")
        with uproot.open(f"{centauro_output_file}:jets") as f:
            centauro_jets = f.arrays(filter_name=["px", "py", "pz", "E", "pt"])
        print("Calculating Centauro quantities")
        centauro_pt = centauro_jets["pt"]
        Q = np.sqrt(Q2)

        centauro_zjet = (centauro_jets["E"] - centauro_jets["pz"]) / Q[:, None]
        zjet_mask = centauro_zjet > flags.centauro_zjetmin
        centauro_pt = centauro_pt[zjet_mask]
        centauro_zjet = centauro_zjet[zjet_mask]

        lab_zjet = ak.Array(jet_output["zjet"])

        leading_centauro = ak.max(centauro_zjet, axis=1, mask_identity=True)
        leading_lab = ak.max(lab_zjet, axis=1, mask_identity=True)
        Delta_z = ak.to_numpy(ak.fill_none(leading_lab - leading_centauro, np.nan))[:, None]

        jet_output["Delta_zjet"] = Delta_z.tolist()
        jet_output["zjet_centauro"] = centauro_zjet.to_list()
        jet_output["jet_centauro_pt"] = centauro_pt.to_list()

    # ── Write output ───────────────────────────────────────────────────
    print("Fraction of skipped events due to electron identification:", unclear_electron_events/n_events)
    print(f"Writing output to {flags.output}")

    # Event-level tree
    event_out = {"Q2": Q2, "W": W, "x": x, "y": y}
    with uproot.recreate(flags.output) as fout:
        fout["events"] = event_out
        fout["jets"] = jet_output

    print("Done.")


if __name__ == "__main__":
    main()
