import numpy as np

import os
import h5py as h5
import glob
import argparse
import uproot
import awkward as ak
def convert_to_np(
    file_list,
    flags,
    max_nonzero_particles=132,  # Maximum number after applying the per-particle selection
    nevts=100000000,
):
    gen_dict = {
        "event_features": [],
        "particle_features": [],
    }
    for ifile, file_name in enumerate(file_list):
        try:
            with uproot.open(file_name) as f:
                particles = f["particles"].arrays(entry_stop = nevts)
                events = f["events"].arrays(entry_stop = nevts)
                electrons = f["electron"].arrays(entry_stop = nevts)

        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            continue


        max_number_of_particles = ak.max(ak.num(particles["pid"], axis=1))
        print(max_number_of_particles)
        y = events["y"]
        fiducial_mask = (y > 0.2) & (y < 0.7) & (events["Q2"] > 150)
        events = events[fiducial_mask]
        particles = particles[fiducial_mask]


        pid = particles["pid"]
        status = particles["status"]
        px = particles["px"]
        py = particles["py"]
        pz = particles["pz"]

        pt = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        theta = np.arccos(pz/np.sqrt(px**2 + py**2 + pz**2))
        eta = -np.log(np.tan(theta/2))

        final_mask = (status > 0)
        electron_px = electrons["px"]
        electron_py = electrons["py"]
        electron_pz = electrons["pz"]

        # Exclude the scattered electron from jet clustering and make our fiducial cuts
        hfs_mask = (final_mask) & (pt > 0.1) & (eta > -1.5) & (eta < 2.75)
        pt = pt[hfs_mask]
        phi = phi[hfs_mask]
        eta = eta[hfs_mask]

        pt_sorting = ak.argsort(pt, ascending=False)
        pt = pt[pt_sorting]
        phi = phi[pt_sorting]
        eta = eta[pt_sorting]

        if max_nonzero_particles < ak.max(ak.num(pt, axis=1)):
            print("WARNING: max_nonzero_particles is smaller than the actual maximum number of particles present")
        
        # Padding the particle arrays to max_nonzero_particles
        # max_nonzero_particles should be the max particle number after the fiducial cuts
        pt = ak.fill_none(ak.pad_none(pt, max_nonzero_particles, clip=True), 0)
        phi = ak.fill_none(ak.pad_none(phi, max_nonzero_particles, clip=True), 0)
        eta = ak.fill_none(ak.pad_none(eta, max_nonzero_particles, clip=True), 0)

        stacked_event_features = np.stack([events["Q2"].to_numpy(), events["y"].to_numpy(), electron_px.to_numpy(), electron_py.to_numpy(), electron_pz.to_numpy(), events["weight"].to_numpy()], -1)
        gen_dict["event_features"].append(stacked_event_features)


        stacked_particle_features = np.stack([pt.to_numpy(), eta.to_numpy(), phi.to_numpy()], -1)
        gen_dict["particle_features"].append(stacked_particle_features)


    gen_dict["event_features"] = np.concatenate(gen_dict["event_features"])
    gen_dict["particle_features"] = np.concatenate(gen_dict["particle_features"])

    return gen_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory",
        default="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_files/",
        help="Folder containing the clustered Pythia files in .root format",
    )
    parser.add_argument(
        "--output_directory",
        default="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/",
        help="Output folder containing Pythia h5 files",
    )

    parser.add_argument(
        "--input_string",
        default="pythia_H1_alphaS14_eplus_10mil",
        help="Sample to process",
    )
    parser.add_argument(
        "--lepton_beam",
        type=int,
        default=11,
        help="PDG number of lepton beam (11 or -11)"
    )
    flags = parser.parse_args()

    print("Processing ", flags.input_string)
    file_list = glob.glob(os.path.join(flags.input_directory, flags.input_string+"*.root"))
    print("Using files: ", file_list)
    gen = convert_to_np(
        file_list,
        flags=flags
    )
    os.makedirs(flags.output_directory, exist_ok=True)
    with h5.File(
        os.path.join(flags.output_directory, f"{flags.input_string}.h5"), "w"
    ) as fh5:
        fh5.create_dataset("gen_particle_features", data=gen["particle_features"])
        fh5.create_dataset("gen_event_features", data=gen["event_features"])
