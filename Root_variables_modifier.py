import ROOT

# Open the input file
input_file = ROOT.TFile.Open("/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root", "READ")

# Get the input tree
input_tree = input_file.Get("Tuple/DecayTree;1")


output_file = ROOT.TFile.Open("/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root", "RECREATE")
output_tree = ROOT.TTree("Tuple/DecayTree", "My tree with modified branch names")

        
for branch in input_tree.GetListOfBranches():
    old_name = branch.GetName()
    new_name = old_name.replace("L1", "ep")
    new_name = new_name.replace("L2", "em")  # Modify the variable name as needed
    branch.SetName(new_name)
    branch_def = branch.GetTitle()
    branch_def = branch_def.replace(old_name, new_name)
    output_tree.Branch(new_name, ROOT.AddressOf(branch), branch_def)


for i in range(input_tree.GetEntries()):
    input_tree.GetEntry(i)
    output_tree.Fill()
    if i < 5:  # Print the values for the first five events
        print("L1:", input_tree.L1_PT, "L2:", input_tree.L2_PT)
        print("ep:", output_tree.ep_PT, "em:", output_tree.em_PT)

print(input_tree.GetEntries())
print(output_tree.GetEntries())

# Write the new TTree to the output file
output_file.WriteTObject(output_tree)

# Close the files
input_file.Close()
output_file.Close()

